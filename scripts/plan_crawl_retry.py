"""Select an unresolved crawl or publication failure for the delayed schedules.

Read the small selection-audit artifacts, not the multi-megabyte recovery bundles.
Hourly checks honor network cooldowns and the actual paid-request time window.
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
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_policy import next_off_peak, scheduled_guard_enabled


DAILY_SCHEDULE = '37 17 * * *'
RETRY_SCHEDULES = {'17 * * * *', '17 19 * * *', '17 21 * * *'}
RETENTION_SECONDS = 14 * 24 * 3600
MAX_COUNT_RETRIES = 3


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def read_audits(artifact):
    archive = subprocess.check_output([
        'gh', 'api', f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/artifacts/{artifact["id"]}/zip'])
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        if artifact['name'].startswith('discovery-audit-'):
            row = json.loads(bundle.read('resume-plan.json'))
            return [dict(status='deferred', target_date='', retryable=True, cooldown_scope='date',
                         retry_not_before=row['retry_not_before'], discovery_pending=True)]
        return [json.loads(bundle.read(name)) for name in bundle.namelist()
                if re.fullmatch(r'(?:run_metrics/)?(?:\d{4}-\d{2}-\d{2}|latest)-(?:crawl(?:-progress)?|pipeline)\.json', name)]


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


def choose_retry(artifacts, load_audits, now, requested_date=None, published_dates=(), decision=None):
    seen, pending = set(), []
    count_failures, closed, count_candidates = {}, set(), set()
    cooldown = 0
    history = []
    for artifact in sorted(artifacts, key=lambda item: item['created_at'], reverse=True):
        if (artifact.get('expired') or not artifact['name'].startswith(('selection-audit-', 'discovery-audit-'))
                or artifact.get('workflow_run', {}).get('head_branch') != 'main'
                or now - timestamp(artifact['created_at']) > RETENTION_SECONDS):
            continue
        # Fail closed if an audit cannot be read. Do not revive an older failure
        # when a more recent success might be in the unavailable artifact.
        audits = load_audits(artifact)
        for audit in sorted(audits, key=lambda item: (item.get('audit_type') != 'pipeline', item.get('status') != 'complete')):
            day = audit.get('target_date') or ''
            if day:
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
                    raise ValueError('Invalid announcement date in crawl audit')
                age = (datetime.fromtimestamp(now, timezone.utc).date() - date.fromisoformat(day)).days
                if not 0 <= age <= 14:
                    continue
            if audit.get('audit_type') != 'pipeline':
                history.append((timestamp(artifact['created_at']), audit))
            if day not in closed:
                if audit.get('status') == 'complete' and audit.get('audit_type') != 'pipeline':
                    closed.add(day)
                elif audit.get('status') == 'failed' and count_mismatch(audit):
                    count_failures[day] = count_failures.get(day, 0) + 1
            if audit.get('status') == 'complete' and audit.get('discovered_latest', True):
                seen.add('')  # A dated success resolves earlier discovery failures.
            if day in seen:
                continue
            seen.add(day)
            if audit.get('status') in {'complete', 'ready', 'running'}:
                if day and day not in published_dates and audit.get('unique_papers') != 0:
                    # The crawl audit precedes report publication. A failed
                    # later step leaves a recoverable, unpublished date.
                    pending.append((day, timestamp(artifact['created_at']), 0))
                continue
            if audit.get('status') not in {'failed', 'deferred'} or not transient(audit):
                continue
            deadline = audit.get('retry_not_before') or timestamp(artifact['created_at']) + 1800
            if audit.get('cooldown_scope', 'date' if count_mismatch(audit) else 'host') == 'host':
                cooldown = max(cooldown, deadline)
            if day not in published_dates:
                pending.append((day, timestamp(artifact['created_at']), deadline))
                if count_mismatch(audit):
                    count_candidates.add(day)
    # Consecutive network failures across runners wait 1, 2, 4, then 8 hours.
    # A verified crawl resets this streak. Retry-After can extend it further.
    streak, latest_failure = 0, 0
    for attempted, audit in history:
        if audit.get('status') == 'complete':
            break
        if (audit.get('status') == 'failed' and transient(audit)
                and audit.get('cooldown_scope', 'date' if count_mismatch(audit) else 'host') == 'host'):
            streak += 1
            latest_failure = latest_failure or attempted
    if streak:
        cooldown = max(cooldown, latest_failure + 3600 * 2 ** min(streak - 1, 3))
    # A server cooldown applies to the host, including retries for other dates.
    if cooldown > now:
        print(f'arXiv host cooldown active until {datetime.fromtimestamp(cooldown, timezone.utc).isoformat()}; deferring crawl.')
        if decision is not None:
            decision.update(reason='host_cooldown', retry_not_before=cooldown, network_failure_streak=streak)
        return None
    if requested_date is not None:
        if any(day == requested_date and deadline > now for day, _, deadline in pending):
            print(f'Announcement cooldown active for {requested_date}; deferring crawl.')
            if decision is not None:
                decision.update(reason='date_cooldown', retry_not_before=max(deadline for day, _, deadline in pending if day == requested_date))
            return None
        return requested_date
    eligible = [(last_attempt, day) for day, last_attempt, deadline in pending
                if deadline <= now and (day not in count_candidates
                                        or count_failures.get(day, 0) <= MAX_COUNT_RETRIES)]
    for day in sorted(day for day in count_candidates if count_failures.get(day, 0) > MAX_COUNT_RETRIES):
        print(f'Count mismatch retry limit reached for {day}; manual investigation required.')
    # Least recently attempted first: one stubborn date cannot consume both
    # retry slots every day while another recoverable date waits indefinitely.
    if not eligible and decision is not None:
        future = [deadline for _, _, deadline in pending if deadline > now]
        decision.update(reason='date_cooldown' if future else 'no_pending_report',
                        retry_not_before=min(future) if future else None)
    return min(eligible)[1] if eligible else None


def plan(event, requested_date, artifacts=None, now=None, published_dates=(), decision=None):
    schedule = event.get('schedule')
    automated = bool(schedule or 'workflow_run' in event)
    retry = schedule in RETRY_SCHEDULES or 'workflow_run' in event
    now = time.time() if now is None else now
    # Check the real start time, not the cron's nominal hour. No arXiv requests
    # or audit downloads are needed while the paid-request window is closed.
    if automated and scheduled_guard_enabled():
        allowed_at = next_off_peak(datetime.fromtimestamp(now, timezone.utc)).timestamp()
        if allowed_at > now:
            if decision is not None:
                decision.update(reason='off_peak_window', retry_not_before=allowed_at,
                                discovery_pending=schedule == DAILY_SCHEDULE)
            print(f'Automatic processing deferred until {datetime.fromtimestamp(allowed_at, timezone.utc).isoformat()}')
            return None
    if artifacts is None:
        repository = os.environ['GITHUB_REPOSITORY']
        pages = json.loads(subprocess.check_output([
            'gh', 'api', '--paginate', '--slurp',
            f'repos/{repository}/actions/artifacts?per_page=100'], text=True))
        artifacts = [artifact for page in pages for artifact in page['artifacts']]
        if retry:
            published_dates = read_published_dates()
    return choose_retry(artifacts, read_audits, now,
                        requested_date=None if retry else requested_date,
                        published_dates=published_dates, decision=decision)


def main():
    with open(os.environ['GITHUB_EVENT_PATH']) as stream:
        event = json.load(stream)
    decision = {}
    selected = plan(event, event.get('inputs', {}).get('target_date', '') or '', decision=decision)
    if selected and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', selected):
        raise ValueError('Invalid requested announcement date')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'should_run={str(selected is not None).lower()}\n')
        output.write(f'target_date={selected or ""}\n')
        output.write(f'discovery_pending={str(decision.get("discovery_pending", False)).lower()}\n')
    decision.update(should_run=selected is not None, target_date=selected,
                    event=os.getenv('GITHUB_EVENT_NAME'), schedule=event.get('schedule'),
                    recorded_at=datetime.now(timezone.utc).isoformat())
    decision.setdefault('reason', 'resume_pending' if selected is not None else 'no_pending_report')
    Path('resume-plan.json').write_text(json.dumps(decision, indent=2) + '\n')
    message = ('No eligible crawl retry; skipping.' if selected is None else
               f'Crawl planned for {selected or "latest announcement"}.')
    print(message)
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
        summary.write(message + '\n')
        summary.write('```json\n' + json.dumps(decision, indent=2) + '\n```\n')


if __name__ == '__main__':
    main()
