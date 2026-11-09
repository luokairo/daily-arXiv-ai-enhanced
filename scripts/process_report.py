"""Stage a complete report, or checkpoint a low-peak deferral without publishing."""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_state(path, day, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(audit_type='pipeline', target_date=day,
               recorded_at=datetime.now(timezone.utc).isoformat(), **fields)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def process(day, data, stage, run=None):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
        raise ValueError('Processing needs a YYYY-MM-DD announcement date')
    run = run or subprocess.run
    data, stage = Path(data).resolve(), Path(stage).resolve()
    state = stage / 'run_metrics' / f'{day}-pipeline.json'
    enhanced = stage / f'{day}_AI_enhanced_{os.getenv("LANGUAGE", "Chinese")}.jsonl'
    commands = [
        ('enhance', ROOT / 'ai', ['enhance.py', '--data', str(data / f'{day}.jsonl')]),
        ('convert', ROOT / 'to_md', ['convert.py', '--data', str(enhanced)]),
    ]
    if os.getenv('ENABLE_DEEP_READ', 'true').lower() in {'1', 'true', 'yes'}:
        commands.append(('deep_read', ROOT / 'ai', ['deep_read.py', '--data', str(enhanced),
            '--output', str(stage / 'deep_reads' / f'{day}.md'),
            '--output-json', str(stage / 'deep_reads' / f'{day}.json')]))
    for phase, cwd, command in commands:
        write_state(state, day, status='running', phase=phase, retryable=True)
        env = {**os.environ, 'TARGET_DATE': day, 'AI_OUTPUT_DIR': str(stage),
               'PIPELINE_STATE_PATH': str(state), 'PIPELINE_PHASE': phase,
               'AI_METRICS_PATH': str(stage / ('deep-read-usage.json' if phase == 'deep_read' else 'enhance-usage.json'))}
        result = run([sys.executable, *command], cwd=cwd, env=env)
        if result.returncode:
            row = json.loads(state.read_text())
            if row.get('status') == 'deferred' and row.get('phase') == phase and row.get('error_type') == 'OffPeakWindowClosed':
                print(f'Report deferred for {day}; phase={phase}; retry_not_before={row["retry_not_before"]}', flush=True)
                return False
            write_state(state, day, status='failed', phase=phase, retryable=False,
                        error_type='ProcessingError', returncode=result.returncode)
            raise subprocess.CalledProcessError(result.returncode, command)
    # A ready pipeline still needs a successful data-branch push to be complete.
    if not enhanced.is_file() or not (stage / f'{day}.md').is_file():
        write_state(state, day, status='failed', phase='validate', retryable=False,
                    error_type='MissingReportArtifacts')
        raise RuntimeError('Processing did not produce both enhanced JSONL and Markdown')
    write_state(state, day, status='ready', phase='publish', retryable=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--data', default='data')
    parser.add_argument('--stage', default=os.getenv('AI_OUTPUT_DIR', '/tmp/arxiv-stage'))
    args = parser.parse_args()
    ready = process(args.date, args.data, args.stage)
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'report_ready={str(ready).lower()}\n')
    message = f'{args.date}: ' + ('report staged; publication pending.' if ready else 'waiting for the next off-peak window; report not published.')
    print(message)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(message + '\n')


if __name__ == '__main__':
    main()
