import unittest
from datetime import datetime, timezone

from scripts.plan_crawl_retry import DAILY_SCHEDULE, RETRY_SCHEDULES, choose_retry, plan


NOW = datetime(2026, 11, 2, tzinfo=timezone.utc).timestamp()


def artifact(number, day='2026-11-01', branch='main'):
    return dict(id=number, name=f'selection-audit-{number}', expired=False,
                created_at=f'{day}T00:00:00Z', workflow_run=dict(head_branch=branch))


def failed(day='2026-10-30', **extra):
    return dict(status='failed', target_date=day, error_type='ReadTimeout', **extra)


class RetryPlanTests(unittest.TestCase):
    def select(self, artifacts, audits, now=NOW):
        return choose_retry(artifacts, lambda item: audits[item['id']], now)

    def test_retries_legacy_timeout_and_rate_limit(self):
        for audit in (failed(), dict(status='failed', target_date='2026-10-30',
                                    error_type='HTTPError', error='429 Client Error')):
            self.assertEqual(self.select([artifact(1)], {1: [audit]}), '2026-10-30')

    def test_success_closes_failure_even_when_snapshot_retrieved_earlier(self):
        audits = {1: [failed()], 2: [dict(status='complete', target_date='2026-10-30',
                                         retrieved_at='2026-10-30T00:00:00Z')]}
        self.assertIsNone(self.select([artifact(1, '2026-10-31'), artifact(2)], audits))

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
                                    {1: [failed(None)], 2: [dict(status='complete', target_date='2026-10-30')]}))

    def test_daily_and_manual_runs_are_unaffected_and_empty_retry_skips(self):
        self.assertEqual(plan({'schedule': DAILY_SCHEDULE}, ''), '')
        self.assertEqual(plan({}, '2026-10-30'), '2026-10-30')
        for schedule in RETRY_SCHEDULES:
            self.assertIsNone(plan({'schedule': schedule}, '', artifacts=[], now=NOW))

    def test_unreadable_audit_must_not_revive_an_older_failure(self):
        def broken(_):
            raise ValueError('corrupt audit')
        with self.assertRaises(ValueError):
            choose_retry([artifact(1)], broken, NOW)


if __name__ == '__main__':
    unittest.main()
