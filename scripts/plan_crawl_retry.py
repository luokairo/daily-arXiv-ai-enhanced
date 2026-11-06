"""Select an unresolved crawl or publication failure for the delayed schedules.

Read the small selection-audit artifacts, not the multi-megabyte recovery bundles.
Two daily retry slots bound traffic and runner usage; no runner sleeps for hours.
Use the newest audit and the published files to determine whether a date is done.
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


def read_published_dates():
    ref = 'refs/heads/data'
    if not subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', ref]):
        return set()
    # Fetch tree metadata only. The planner does not need report or cache blobs.
    subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', '--filter=blob:none', 'origin', ref], check=True)
    if not subprocess.check_output(['git', 'ls-tree', '-d', 'FETCH_HEAD', '--', 'data']):
        return set()
    entries = subprocess.check_output(['git', 'ls-tree', '-z', 'FETCH_HEAD:data'], text=True)
    files = set()
    for entry in entries.split('\0')[:-1]:
        attributes, name = entry.split('\t', 1)
        if attributes.split()[1] == 'blob':
            files.add(name)
    return {match[1] for name in files
            if (match := re.fullmatch(r'(\d{4}-\d{2}-\d{2})_AI_enhanced_.+\.jsonl', name))
            and f'{match[1]}.md' in files}


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


def choose_retry(artifacts, load_audits, now, requested_date=None, published_dates=()):
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
                if day and day not in published_dates and audit.get('unique_papers') != 0:
                    # The crawl audit precedes report publication. A failed
                    # later step leaves a recoverable, unpublished date.
                    pending.append((day, timestamp(artifact['created_at']), 0))
                continue
            if audit.get('status') != 'failed' or not transient(audit):
                continue
            deadline = audit.get('retry_not_before') or timestamp(artifact['created_at']) + 1800
            if audit.get('cooldown_scope', 'date' if count_mismatch(audit) else 'host') == 'host':
                cooldown = max(cooldown, deadline)
            if day not in published_dates:
                pending.append((day, timestamp(artifact['created_at']), deadline))
                if count_mismatch(audit):
                    count_candidates.add(day)
    # A server cooldown applies to the host, including retries for other dates.
    if cooldown > now:
        print(f'arXiv host cooldown active until {datetime.fromtimestamp(cooldown, timezone.utc).isoformat()}; deferring crawl.')
        return None
    if requested_date is not None:
        if any(day == requested_date and deadline > now for day, _, deadline in pending):
            print(f'Announcement cooldown active for {requested_date}; deferring crawl.')
            return None
        return requested_date
    eligible = [(last_attempt, day) for day, last_attempt, deadline in pending
                if deadline <= now and (day not in count_candidates
                                        or count_failures.get(day, 0) <= MAX_COUNT_RETRIES)]
    for day in sorted(day for day in count_candidates if count_failures.get(day, 0) > MAX_COUNT_RETRIES):
        print(f'Count mismatch retry limit reached for {day}; manual investigation required.')
    # Least recently attempted first: one stubborn date cannot consume both
    # retry slots every day while another recoverable date waits indefinitely.
    return min(eligible)[1] if eligible else None


def plan(event, requested_date, artifacts=None, now=None, published_dates=()):
    schedule = event.get('schedule')
    if artifacts is None:
        repository = os.environ['GITHUB_REPOSITORY']
        pages = json.loads(subprocess.check_output([
            'gh', 'api', '--paginate', '--slurp',
            f'repos/{repository}/actions/artifacts?per_page=100'], text=True))
        artifacts = [artifact for page in pages for artifact in page['artifacts']]
        if schedule in RETRY_SCHEDULES:
            published_dates = read_published_dates()
    return choose_retry(artifacts, read_audits, time.time() if now is None else now,
                        requested_date=None if schedule in RETRY_SCHEDULES else requested_date,
                        published_dates=published_dates)


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
