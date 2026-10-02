"""Save/restore successful AI cache and dated raw snapshots, never partial reports."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def copy_tree(source, destination):
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def save_bundle(data, stage, output, date):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('Recovery needs a verified YYYY-MM-DD date')
    output.mkdir(parents=True, exist_ok=True)
    copy_tree(data / 'ai_cache', output / 'data/ai_cache')
    raw = data / f'{date}.jsonl'
    if raw.exists():
        (output / 'data').mkdir(exist_ok=True)
        shutil.copy2(raw, output / 'data' / raw.name)
    copy_tree(data / 'run_metrics', output / 'data/run_metrics')
    copy_tree(stage / 'run_metrics', output / 'data/run_metrics')
    (output / 'usage').mkdir(exist_ok=True)
    for usage in stage.glob('*usage.json'):
        shutil.copy2(usage, output / 'usage' / usage.name)
    if (stage / 'deep_reads/deep-read-progress.json').exists():
        shutil.copy2(stage / 'deep_reads/deep-read-progress.json', output / 'usage/deep-read-progress.json')
    manifest = dict(version=1, date=date, files={
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'recovery.json'})
    (output / 'recovery.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def restore_zip(archive, data):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read('recovery.json'))
        if manifest.get('version') != 1:
            raise ValueError('Unsupported recovery version')
        for name, digest in manifest['files'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or name not in names:
                raise ValueError('Invalid recovery path')
            content = bundle.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Recovery checksum mismatch')
            # Cache keys/schema are validated again by CachedChain before use.
            if name.startswith('data/ai_cache/') or name.startswith('data/run_metrics/') or re.fullmatch(r'data/\d{4}-\d{2}-\d{2}\.jsonl', name):
                destination = data / path.relative_to('data')
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    return manifest['date']


def restore_remote(data, repository, date=''):
    if date and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('Invalid recovery date')
    response = subprocess.check_output(['gh', 'api', '--paginate', '--slurp',
        f'repos/{repository}/actions/artifacts?per_page=100'], text=True)
    artifacts = [a for page in json.loads(response) for a in page['artifacts']
                 if not a['expired'] and a['name'].startswith('ai-recovery-' + (date + '-' if date else ''))]
    artifacts.sort(key=lambda a: a['created_at'], reverse=True)
    restored = 0
    for artifact in artifacts[:3]:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'recovery.zip'
                with archive.open('wb') as stream:
                    subprocess.run(['gh', 'api', f'repos/{repository}/actions/artifacts/{artifact["id"]}/zip'], stdout=stream, check=True)
                recovered_date = restore_zip(archive, data)
                restored += 1
                print(f'Restored successful cache/snapshot from {artifact["name"]} ({recovered_date})', flush=True)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            print(f'Skipped unusable recovery artifact: {type(error).__name__}', flush=True)
    print(f'Recovery artifacts restored: {restored}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['save', 'restore'])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--stage', type=Path, default=Path('/tmp/arxiv-stage'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/arxiv-recovery'))
    parser.add_argument('--date', default='')
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY', ''))
    args = parser.parse_args()
    if args.mode == 'save':
        save_bundle(args.data, args.stage, args.output, args.date)
    else:
        restore_remote(args.data, args.repository, args.date)


if __name__ == '__main__':
    main()
