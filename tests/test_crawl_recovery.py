import json
import os
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import crawl_dated as crawl
from scripts.recovery import restore_zip, save_bundle
from test_dated_crawl import catchup, listing


DAY = '2026-09-30'
URL = f'https://arxiv.org/catchup/cs.CV/{DAY}?abs=True&page=1'


def response(status=200, header='', body='ok'):
    result = crawl.requests.Response()
    result.status_code = status
    result.headers['Retry-After'] = header
    result._content = body.encode()
    return result


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.sleep = Mock(side_effect=self.advance)
        for target, settings in [
                ('time.monotonic', dict(side_effect=lambda: self.now)),
                ('time.sleep', dict(new=self.sleep)),
                ('random.uniform', dict(return_value=0))]:
            patcher = patch.object(crawl.time if target.startswith('time.') else crawl.random,
                                   target.split('.')[1], **settings)
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.now += seconds

    def test_backoff_recovers_429_timeout_and_503(self):
        client = crawl.Client()
        client.session.get = Mock(side_effect=[response(429), crawl.requests.ReadTimeout('slow'),
                                               response(503), response()])
        self.assertEqual(client.get(URL), 'ok')
        self.assertEqual(client.session.get.call_count, 4)
        for seconds in (60, 120, 240):
            self.assertIn(unittest.mock.call(seconds), self.sleep.call_args_list)

    def test_retry_after_seconds_and_http_date_are_respected(self):
        now = datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc).timestamp()
        for header in ('180', 'Wed, 07 Oct 2026 00:03:00 GMT'):
            with self.subTest(header=header), patch.object(crawl.time, 'time', return_value=now):
                client = crawl.Client()
                client.session.get = Mock(side_effect=[response(429, header), response()])
                self.assertEqual(client.get(URL), 'ok')
                self.assertIn(unittest.mock.call(180), self.sleep.call_args_list)
        for header in ('nonsense', '-1', 'Tue, 06 Oct 2026 00:00:00 GMT', ''):
            with patch.object(crawl.time, 'time', return_value=now):
                self.assertEqual(crawl.retry_after_seconds(header), 0)

    def test_long_server_cooldown_is_not_shortened(self):
        client = crawl.Client()
        client.session.get = Mock(return_value=response(429, '1200'))
        with self.assertRaises(crawl.requests.HTTPError):
            client.get(URL)
        self.assertEqual(client.session.get.call_count, 1)
        self.assertFalse(any(call.args[0] > 0 for call in self.sleep.call_args_list))

    def test_timeouts_consume_retry_budget_and_keep_failed_url(self):
        client = crawl.Client(retry_budget=300)
        def timeout(*args, **kwargs):
            self.advance(90)
            raise crawl.requests.ReadTimeout('slow')
        client.session.get = Mock(side_effect=timeout)
        with self.assertRaises(crawl.requests.ReadTimeout):
            client.get(URL)
        self.assertEqual(client.session.get.call_count, 2)
        self.assertLessEqual(self.now - 1000, 300)
        self.assertEqual(client.current_url, URL)

    def test_terminal_http_errors_and_minimum_request_spacing(self):
        client = crawl.Client()
        client.session.get = Mock(return_value=response(403))
        with self.assertRaises(crawl.requests.HTTPError):
            client.get(URL)
        self.assertEqual(client.session.get.call_count, 1)
        client.session.get = Mock(return_value=response())
        client.get(URL)
        client.get(URL)
        self.assertGreaterEqual(self.now - 1000, 30.2)


@patch.object(crawl, 'recall_categories', return_value=['cs.CV'])
class CrawlRecoveryTests(unittest.TestCase):
    def test_interrupted_crawl_roundtrip_resumes_only_missing_page(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, stage = root / 'data', root / 'stage'
            output = root / 'github-output'
            client = Mock()
            client.get.side_effect = [catchup(expected=2, total=2), crawl.requests.ReadTimeout('page 2')]
            with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
                with self.assertRaises(crawl.requests.ReadTimeout):
                    crawl.crawl(DAY, data, stage / 'run_metrics', client)
            self.assertEqual(output.read_text(), f'crawl_date={DAY}\n')
            self.assertFalse((data / f'{DAY}.jsonl').exists())
            self.assertFalse((stage / f'run_metrics/{DAY}-crawl.json').exists())
            # Half-built model/Markdown outputs cannot be recovered for publication.
            (stage / f'{DAY}.md').write_text('partial report')
            bundle = root / 'bundle'
            manifest = save_bundle(data, stage, bundle, DAY)
            self.assertIn(f'data/crawl_cache/{DAY}/cs.CV/page-1.json', manifest['files'])
            archive = root / 'bundle.zip'
            with zipfile.ZipFile(archive, 'w') as zip_file:
                for path in bundle.rglob('*'):
                    if path.is_file():
                        zip_file.write(path, str(path.relative_to(bundle)))
            restored = root / 'restored'
            restore_zip(archive, restored)
            self.assertFalse((restored / f'{DAY}.md').exists())
            client.get = Mock(return_value=catchup(ids=('2609.10002',), expected=2, total=2))
            result = crawl.crawl(DAY, restored, restored / 'run_metrics', client)
            self.assertEqual(result['unique_papers'], 2)
            client.get.assert_called_once_with(URL.replace('page=1', 'page=2'))

    def test_invalid_expired_or_wrong_date_cache_is_refetched(self, _):
        with tempfile.TemporaryDirectory() as temp:
            cache = crawl.PageCache(Path(temp), DAY, 'cs.CV')
            for field, value in [('sha256', 'wrong'), ('day', '2026-09-29'),
                                 ('url', 'https://example.com'), ('retrieved_at', 0),
                                 ('html', 123), ('schema_version', 2)]:
                with self.subTest(field=field):
                    cache.save(1, URL, catchup())
                    path = cache.directory / 'page-1.json'
                    entry = json.loads(path.read_text())
                    entry[field] = value
                    path.write_text(json.dumps(entry))
                    client = Mock(get=Mock(return_value=catchup()))
                    papers, _ = crawl.category_catchup(client, 'cs.CV', DAY, Path(temp))
                    self.assertEqual(len(papers), 1)
                    client.get.assert_called_once_with(URL)
            cache.save(1, URL, catchup(day='2026-09-29'))
            self.assertIsNone(cache.load(1, URL))  # Valid hash alone is insufficient.

    def test_changed_pagination_discards_cache_and_fetches_fresh_once(self, _):
        with tempfile.TemporaryDirectory() as temp:
            cache = crawl.PageCache(Path(temp), DAY, 'cs.CV')
            cache.save(1, URL, catchup(expected=2, total=2))
            client = Mock()
            client.get.side_effect = [catchup(ids=('2609.10002',), expected=3, total=3),
                                     catchup(ids=('2609.10001', '2609.10002'), expected=3, total=3),
                                     catchup(ids=('2609.10003',), expected=3, total=3)]
            papers, _ = crawl.category_catchup(client, 'cs.CV', DAY, Path(temp))
            self.assertEqual(len(papers), 3)
            self.assertEqual(client.get.call_count, 3)
            self.assertEqual(client.get.call_args_list[1].args[0], URL)

    def test_blank_date_reuses_complete_snapshot_after_discovery(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            crawl.crawl(DAY, root, root / 'run_metrics', Mock(get=Mock(return_value=catchup())))
            client = Mock(get=Mock(return_value=listing()))
            result = crawl.crawl('', root, root / 'run_metrics', client)
            self.assertEqual(result['unique_papers'], 1)
            client.get.assert_called_once()
            self.assertIn('/pastweek?', client.get.call_args.args[0])

    def test_discovery_failure_has_audit_without_success_output(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock(get=Mock(side_effect=crawl.requests.ReadTimeout('listing')))
            with patch.dict(os.environ, GITHUB_OUTPUT=str(root / 'output')):
                with self.assertRaises(crawl.requests.ReadTimeout):
                    crawl.crawl('', root, root / 'audit', client)
            audit = json.loads((root / 'audit/latest-crawl-progress.json').read_text())
            self.assertIsNone(audit['target_date'])
            self.assertFalse((root / 'output').exists())

    def test_corrupt_snapshot_is_replaced_only_after_complete_crawl(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'run_metrics').mkdir()
            (root / f'{DAY}.jsonl').write_text('previous corrupt snapshot')
            (root / f'run_metrics/{DAY}-crawl.json').write_text('malformed JSON')
            client = Mock(get=Mock(return_value=catchup()))
            result = crawl.crawl(DAY, root, root / 'run_metrics', client)
            self.assertEqual(result['unique_papers'], 1)
            client.get.assert_called_once()

    def test_cross_category_version_conflict_preserves_output_and_drops_pages(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / f'{DAY}.jsonl'
            previous.write_text('previous successful output')
            client = Mock(get=Mock(side_effect=[catchup(), catchup(summary='Changed abstract')]))
            with patch.object(crawl, 'recall_categories', return_value=['cs.AI', 'cs.CV']):
                with self.assertRaisesRegex(ValueError, 'Metadata changed across category'):
                    crawl.crawl(DAY, root, root / 'run_metrics', client)
            self.assertEqual(previous.read_text(), 'previous successful output')
            self.assertFalse((root / f'run_metrics/{DAY}-crawl.json').exists())
            self.assertFalse((root / 'crawl_cache' / DAY).exists())


if __name__ == '__main__':
    unittest.main()
