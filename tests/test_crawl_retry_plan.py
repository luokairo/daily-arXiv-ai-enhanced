import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from scripts.plan_crawl_retry import DAILY_SCHEDULE, RETRY_SCHEDULES, choose_retry, plan, read_published_dates


NOW = datetime(2026, 11, 2, tzinfo=timezone.utc).timestamp()


def artifact(number, day='2026-11-01', branch='main'):
    return dict(id=number, name=f'selection-audit-{number}', expired=False,
                created_at=f'{day}T00:00:00Z', workflow_run=dict(head_branch=branch))


def failed(day='2026-10-30', **extra):
    return dict(status='failed', target_date=day, error_type='ReadTimeout', **extra)


class RetryPlanTests(unittest.TestCase):
    def select(self, artifacts, audits, now=NOW, published_dates=()):
        return choose_retry(artifacts, lambda item: audits[item['id']], now, published_dates=published_dates)

    def test_retries_legacy_timeout_and_rate_limit(self):
        for audit in (failed(), dict(status='failed', target_date='2026-10-30',
                                    error_type='HTTPError', error='429 Client Error')):
            self.assertEqual(self.select([artifact(1)], {1: [audit]}), '2026-10-30')

    def test_published_report_closes_failure_even_when_snapshot_retrieved_earlier(self):
        audits = {1: [failed()], 2: [dict(status='complete', target_date='2026-10-30',
                                         retrieved_at='2026-10-30T00:00:00Z')]}
        self.assertIsNone(self.select([artifact(1, '2026-10-31'), artifact(2)], audits,
                                      published_dates={'2026-10-30'}))

    def test_completed_crawl_without_published_report_stays_eligible(self):
        audits = {1: [dict(status='complete', target_date='2026-10-30', unique_papers=557)]}
        self.assertEqual(self.select([artifact(1)], audits), '2026-10-30')
        self.assertIsNone(self.select([artifact(1)], audits, published_dates={'2026-10-30'}))
        audits[1][0]['unique_papers'] = 0
        self.assertIsNone(self.select([artifact(1)], audits))

    def test_published_dates_require_markdown_and_enhanced_jsonl(self):
        files = ['2026-10-28.md', '2026-10-29_AI_enhanced_Chinese.jsonl',
                 '2026-10-30.md', '2026-10-30_AI_enhanced_Chinese.jsonl', '2026-10-29.jsonl']
        tree = ''.join(f'100644 blob oid\t{name}\0' for name in files)
        with patch('scripts.plan_crawl_retry.subprocess.run'), patch(
                'scripts.plan_crawl_retry.subprocess.check_output', side_effect=[b'data ref', b'data tree', tree]):
            self.assertEqual(read_published_dates(), {'2026-10-30'})

    def test_count_mismatch_allows_three_delayed_retries_then_stops(self):
        artifacts = [dict(artifact(i), created_at=f'2026-11-01T0{i}:00:00Z') for i in range(1, 5)]
        audits = {i: [dict(status='failed', target_date='2026-10-30',
                          error_type='CountMismatchError', retryable=True)] for i in range(1, 5)}
        for failures in (1, 2, 3):
            self.assertEqual(self.select(artifacts[:failures], audits), '2026-10-30')
        self.assertIsNone(self.select(artifacts, audits))

    def test_legacy_count_failure_is_eligible_even_with_old_false_flag(self):
        audit = dict(status='failed', target_date='2026-10-30', error_type='ValueError',
                     error='Incomplete catchup day', retryable=False)
        self.assertEqual(self.select([artifact(1)], {1: [audit]}), '2026-10-30')

    def test_count_budget_stops_at_previous_success_and_does_not_block_other_date(self):
        artifacts = [dict(artifact(i), created_at=f'2026-11-01T0{i}:00:00Z') for i in range(1, 7)]
        mismatch = dict(status='failed', target_date='2026-10-30', error_type='CountMismatchError')
        audits = {i: [dict(mismatch)] for i in range(1, 7)}
        audits[5] = [dict(status='complete', target_date='2026-10-30')]
        self.assertEqual(self.select(artifacts, audits), '2026-10-30')
        audits[5] = [dict(mismatch)]
        audits[6] = [failed('2026-10-31')]
        self.assertEqual(self.select(artifacts, audits), '2026-10-31')

    def test_oldest_unresolved_day_survives_success_of_newer_day(self):
        audits = {1: [failed()], 2: [dict(status='complete', target_date='2026-10-31')]}
        self.assertEqual(self.select([artifact(1, '2026-10-31'), artifact(2)], audits), '2026-10-30')

    def test_server_cooldown_blocks_other_dates_until_deadline(self):
        audits = {1: [failed(retryable=True, retry_not_before=NOW + 7200)],
                  2: [failed('2026-10-29')]}
        artifacts = [artifact(1), artifact(2, '2026-10-31')]
        self.assertIsNone(self.select(artifacts, audits))
        self.assertEqual(self.select(artifacts, audits, NOW + 7201), '2026-10-29')

    def test_validation_authentication_and_terminal_errors_are_not_retried(self):
        for audit in (failed(retryable=False),
                      dict(status='failed', target_date='2026-10-30', error_type='ValueError'),
                      dict(status='failed', target_date='2026-10-30', error_type='HTTPError', error='403 Forbidden')):
            self.assertIsNone(self.select([artifact(1)], {1: [audit]}))

    def test_expired_foreign_branch_and_old_announcements_are_ignored(self):
        old = artifact(3, '2026-10-01')
        expired = dict(artifact(4), expired=True)
        self.assertIsNone(self.select([artifact(1, branch='test'), old, expired], {}))
        self.assertIsNone(self.select([artifact(2)], {2: [failed('2026-10-01')]}))

    def test_discovery_failure_retries_latest_until_dated_success(self):
        self.assertEqual(self.select([artifact(1)], {1: [failed(None)]}), '')
        self.assertIsNone(self.select([artifact(1, '2026-10-31'), artifact(2)],
                                    {1: [failed(None)], 2: [dict(status='complete', target_date='2026-10-30')]},
                                    published_dates={'2026-10-30'}))

    def test_daily_and_manual_runs_are_unaffected_and_empty_retry_skips(self):
        self.assertEqual(plan({'schedule': DAILY_SCHEDULE}, '', artifacts=[], now=NOW), '')
        self.assertEqual(plan({}, '2026-10-30', artifacts=[], now=NOW), '2026-10-30')
        for schedule in RETRY_SCHEDULES:
            self.assertIsNone(plan({'schedule': schedule}, '', artifacts=[], now=NOW))

    def test_primary_and_manual_runs_honor_host_cooldown(self):
        audits = {1: [failed(retryable=True, retry_not_before=NOW + 7200)]}
        for requested in ('', '2026-10-31'):
            self.assertIsNone(choose_retry([artifact(1)], lambda a: audits[a['id']], NOW,
                                          requested_date=requested))

    def test_count_cooldown_is_per_date_and_retry_slots_rotate(self):
        artifacts = [artifact(1, '2026-10-31'), artifact(2)]
        audits = {1: [failed('2026-10-31')],
                  2: [dict(status='failed', target_date='2026-10-30', error_type='CountMismatchError',
                           retry_not_before=NOW + 7200, cooldown_scope='date')]}
        self.assertEqual(self.select(artifacts, audits), '2026-10-31')
        audits[2][0]['retry_not_before'] = NOW - 1
        # The older announcement was tried more recently; give the other date a turn.
        self.assertEqual(self.select(artifacts, audits), '2026-10-31')
        self.assertEqual(choose_retry(artifacts, lambda a: audits[a['id']], NOW,
                                     requested_date='2026-10-30'), '2026-10-30')

    def test_unreadable_audit_must_not_revive_an_older_failure(self):
        def broken(_):
            raise ValueError('corrupt audit')
        with self.assertRaises(ValueError):
            choose_retry([artifact(1)], broken, NOW)


if __name__ == '__main__':
    unittest.main()
