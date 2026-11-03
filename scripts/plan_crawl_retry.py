"""Select an unresolved transient crawl failure for the delayed schedules.

Read the small selection-audit artifacts, not the multi-megabyte recovery bundles.
Two daily retry slots bound traffic and runner usage; no runner sleeps for hours.
The newest audit for each announcement wins, so a complete crawl closes its retry.
"""
import io
import json
import os
import re
import subprocess
import time
import zipfile
from datetime import date, datetime, timezone


DAILY_SCHEDULE = '37 17 * * *'
RETRY_SCHEDULES = {'17 19 * * *', '17 21 * * *'}
RETENTION_SECONDS = 14 * 24 * 3600
MAX_COUNT_RETRIES = 3


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def read_audits(artifact):
    archive = subprocess.check_output([
        'gh', 'api', f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/artifacts/{artifact["id"]}/zip'])
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        return [json.loads(bundle.read(name)) for name in bundle.namelist()
                if re.fullmatch(r'(?:run_metrics/)?(?:\d{4}-\d{2}-\d{2}|latest)-crawl(?:-progress)?\.json', name)]


def count_mismatch(audit):
    return (audit.get('error_type') == 'CountMismatchError' or
            (audit.get('error_type') == 'ValueError'
             and audit.get('error') == 'Incomplete catchup day'))


def transient(audit):
    if count_mismatch(audit):
        return True  # Includes the exact legacy error from before typed diagnostics.
    if 'retryable' in audit:
        return audit['retryable'] is True
    # Recovery of failures produced before retry metadata was introduced.
    return (audit.get('error_type') in {'ReadTimeout', 'ConnectTimeout', 'Timeout', 'ConnectionError'}
            or (audit.get('error_type') == 'HTTPError'
                and re.search(r'\b(?:429|5\d\d)\b', audit.get('error', '')) is not None))


def choose_retry(artifacts, load_audits, now):
    seen, pending = set(), []
    count_failures, closed, count_candidates = {}, set(), set()
    cooldown = 0
    for artifact in sorted(artifacts, key=lambda item: item['created_at'], reverse=True):
        if (artifact.get('expired') or not artifact['name'].startswith('selection-audit-')
                or artifact.get('workflow_run', {}).get('head_branch') != 'main'
                or now - timestamp(artifact['created_at']) > RETENTION_SECONDS):
            continue
        # Fail closed if an audit cannot be read. Do not revive an older failure
        # when a more recent success might be in the unavailable artifact.
        audits = load_audits(artifact)
        for audit in sorted(audits, key=lambda item: item.get('status') != 'complete'):
            day = audit.get('target_date') or ''
            if day:
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
                    raise ValueError('Invalid announcement date in crawl audit')
                age = (datetime.fromtimestamp(now, timezone.utc).date() - date.fromisoformat(day)).days
                if not 0 <= age <= 14:
                    continue
            if day not in closed:
                if audit.get('status') == 'complete':
                    closed.add(day)
                elif audit.get('status') == 'failed' and count_mismatch(audit):
                    count_failures[day] = count_failures.get(day, 0) + 1
            if day in seen:
                continue
            seen.add(day)
            if audit.get('status') == 'complete':
                seen.add('')  # A later dated crawl also resolves discovery failures.
                continue
            if audit.get('status') != 'failed' or not transient(audit):
                continue
            deadline = audit.get('retry_not_before') or timestamp(artifact['created_at']) + 1800
            cooldown = max(cooldown, deadline)
            pending.append(day)
            if count_mismatch(audit):
                count_candidates.add(day)
    # A server cooldown applies to the host, including retries for other dates.
    if cooldown > now:
        return None
    eligible = [day for day in pending if day not in count_candidates
                or count_failures.get(day, 0) <= MAX_COUNT_RETRIES]
    for day in sorted(count_candidates - set(eligible)):
        print(f'Count mismatch retry limit reached for {day}; manual investigation required.')
    return min(eligible) if eligible else None


def plan(event, requested_date, artifacts=None, now=None):
    schedule = event.get('schedule')
    if schedule not in RETRY_SCHEDULES:
        return requested_date
    if artifacts is None:
        repository = os.environ['GITHUB_REPOSITORY']
        pages = json.loads(subprocess.check_output([
            'gh', 'api', '--paginate', '--slurp',
            f'repos/{repository}/actions/artifacts?per_page=100'], text=True))
        artifacts = [artifact for page in pages for artifact in page['artifacts']]
    return choose_retry(artifacts, read_audits, time.time() if now is None else now)


def main():
    with open(os.environ['GITHUB_EVENT_PATH']) as stream:
        event = json.load(stream)
    selected = plan(event, event.get('inputs', {}).get('target_date', '') or '')
    if selected and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', selected):
        raise ValueError('Invalid requested announcement date')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'should_run={str(selected is not None).lower()}\n')
        output.write(f'target_date={selected or ""}\n')
    message = ('No eligible crawl retry; skipping.' if selected is None else
               f'Crawl planned for {selected or "latest announcement"}.')
    print(message)
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
        summary.write(message + '\n')


if __name__ == '__main__':
    main()
