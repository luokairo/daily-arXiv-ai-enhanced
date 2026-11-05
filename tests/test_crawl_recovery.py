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

    def test_backoff_recovers_timeout_and_server_errors(self):
        client = crawl.Client()
        client.session.get = Mock(side_effect=[response(503), response(503), response()])
        client.session.close = Mock()
        self.assertEqual(client.get(URL), 'ok')
        self.assertEqual(client.session.get.call_count, 3)
        self.assertEqual(client.session.close.call_count, 2)
        for seconds in (60, 120):
            self.assertIn(unittest.mock.call(seconds), self.sleep.call_args_list)
        client.session.get = Mock(side_effect=[crawl.requests.ReadTimeout('slow'), response()])
        self.assertEqual(client.get(URL), 'ok')

    def test_429_defers_without_repeating_requests_and_cools_for_one_hour(self):
        client = crawl.Client()
        client.session.get = Mock(return_value=response(429))
        before = crawl.time.time()
        with self.assertRaises(crawl.requests.HTTPError):
            client.get(URL)
        self.assertEqual(client.session.get.call_count, 1)
        self.assertGreaterEqual(client.retry_not_before, before + 3600)

    def test_repeated_transport_failures_are_bounded_even_without_budget_pressure(self):
        client = crawl.Client()
        client.session.get = Mock(side_effect=crawl.requests.ConnectionError('disconnected'))
        with self.assertRaises(crawl.requests.ConnectionError):
            client.get(URL)
        self.assertEqual(client.session.get.call_count, 2)

    def test_retry_after_seconds_and_http_date_are_respected(self):
        now = datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc).timestamp()
        for header in ('180', 'Wed, 07 Oct 2026 00:03:00 GMT'):
            with self.subTest(header=header), patch.object(crawl.time, 'time', return_value=now):
                client = crawl.Client()
                client.session.get = Mock(side_effect=[response(503, header), response()])
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
        self.assertGreater(client.retry_not_before, crawl.time.time() + 1190)

    def test_cooldown_survives_the_immediate_retry_budget(self):
        client = crawl.Client()
        client.session.get = Mock(return_value=response(429, '7200'))
        with tempfile.TemporaryDirectory() as temp, patch.object(crawl, 'recall_categories', return_value=['cs.CV']):
            root = Path(temp)
            with self.assertRaises(crawl.requests.HTTPError):
                crawl.crawl(DAY, root, root / 'audit', client)
            audit = json.loads((root / f'audit/{DAY}-crawl-progress.json').read_text())
            self.assertTrue(audit['retryable'])
            self.assertEqual(audit['http_status'], 429)
            self.assertGreater(audit['retry_not_before'], crawl.time.time() + 7190)

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
    def test_metadata_conflict_requires_fresh_pages_then_recovers(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(crawl, 'recall_categories', return_value=['cs.AI', 'cs.CV']):
                client = Mock(get=Mock(side_effect=[catchup(summary='old'), catchup(summary='new'), '<html></html>']))
                with self.assertRaises(crawl.MetadataConflictError):
                    crawl.crawl(DAY, root, root / 'run_metrics', client)
                progress = json.loads((root / f'run_metrics/{DAY}-crawl-progress.json').read_text())
                self.assertTrue(progress['retryable'])
                self.assertEqual(progress['cooldown_scope'], 'date')
                self.assertEqual(progress['metadata_refresh_attempts'], 1)
                self.assertTrue((root / f'crawl_diagnostics/{DAY}/metadata-conflict.json').exists())
                # An older bundle must not reintroduce a known stale category.
                cache = crawl.PageCache(root / 'crawl_cache', DAY, 'cs.AI')
                cache.save(1, URL.replace('cs.CV', 'cs.AI'), catchup(summary='old'),
                           retrieved_at=progress['require_fresh_after'] - 100)
                client = Mock(get=Mock(side_effect=[catchup(summary='new'), catchup(summary='new')]))
                result = crawl.crawl(DAY, root, root / 'run_metrics', client)
                self.assertEqual(result['unique_papers'], 1)
                self.assertEqual(client.get.call_count, 2)

    def test_persistent_fresh_metadata_conflict_stops_after_one_refresh(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(crawl, 'recall_categories', return_value=['cs.AI', 'cs.CV']):
                for attempt in (1, 2):
                    client = Mock(get=Mock(side_effect=[catchup(summary='one'), catchup(summary='two'), '<html></html>']))
                    with self.assertRaises(crawl.MetadataConflictError):
                        crawl.crawl(DAY, root, root / 'run_metrics', client)
                    progress = json.loads((root / f'run_metrics/{DAY}-crawl-progress.json').read_text())
                    self.assertEqual(progress['metadata_refresh_attempts'], attempt)
                    self.assertEqual(progress['retryable'], attempt == 1)

    def test_newest_progress_audit_wins_across_recovery_bundles(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archives = []
            for index, recorded_at in enumerate(('2026-11-03T00:00:00+00:00', '2026-11-05T00:00:00+00:00')):
                data = root / f'data-{index}'
                path = data / f'run_metrics/{DAY}-crawl-progress.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(target_date=DAY, recorded_at=recorded_at, version=index)))
                bundle = root / f'bundle-{index}'
                save_bundle(data, root / 'stage', bundle, DAY)
                archive = root / f'bundle-{index}.zip'
                with zipfile.ZipFile(archive, 'w') as stream:
                    for file in bundle.rglob('*'):
                        if file.is_file():
                            stream.write(file, str(file.relative_to(bundle)))
                archives.append(archive)
            for archive in (archives[0], archives[1], archives[0]):
                restore_zip(archive, root / 'restored')
            progress = json.loads((root / f'restored/run_metrics/{DAY}-crawl-progress.json').read_text())
            self.assertEqual(progress['version'], 1)

    def test_count_failure_preserves_html_counts_and_requires_delayed_retry(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Page total and actual entries agree; section expected count does not.
            html = catchup(expected=2, total=1)
            client = Mock(get=Mock(return_value=html))
            with patch.dict(os.environ, GITHUB_OUTPUT=str(root / 'output')):
                with self.assertRaises(crawl.CountMismatchError):
                    crawl.crawl(DAY, root / 'data', root / 'stage/run_metrics', client)
            client.get.assert_called_once()  # No immediate retry loop on fresh bad HTML.
            progress = json.loads((root / f'stage/run_metrics/{DAY}-crawl-progress.json').read_text())
            self.assertTrue(progress['retryable'])
            self.assertEqual(progress['count_details']['total'], 1)
            self.assertEqual(progress['count_details']['actual'], 1)
            self.assertEqual(progress['count_details']['expected_sum'], 2)
            self.assertGreater(progress['retry_not_before'], crawl.time.time() + 1790)
            diagnostic = json.loads(Path(progress['diagnostic_path']).read_text())
            self.assertEqual(diagnostic['pages'][0]['counts']['sections']['New'], dict(expected=2, actual=1))
            self.assertEqual(Path(progress['diagnostic_path']).with_name('page-1.html').read_text(), html)
            self.assertEqual((root / 'output').read_text(), f'crawl_date={DAY}\n')
            self.assertFalse((root / f'data/{DAY}.jsonl').exists())
            self.assertFalse((root / f'data/crawl_cache/{DAY}/cs.CV').exists())
            # A later run re-fetches the failed category and succeeds fully.
            client.get = Mock(return_value=catchup())
            result = crawl.crawl(DAY, root / 'data', root / 'stage/run_metrics', client)
            self.assertEqual(result['unique_papers'], 1)
            client.get.assert_called_once()
            self.assertTrue(Path(progress['diagnostic_path']).exists())

    def test_section_mismatch_has_declared_and_actual_counts(self, _):
        html = catchup().replace('showing 1 of 1', 'showing 2 of 2')
        with self.assertRaises(crawl.CountMismatchError) as caught:
            crawl.parse_catchup(html, DAY, 'cs.CV')
        self.assertEqual(caught.exception.counts,
                         dict(section='New', declared_shown=2, declared_expected=2, actual=1, total=1))

    def test_cached_failure_diagnostics_survive_successful_fresh_pass(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = crawl.PageCache(root / 'cache', DAY, 'cs.CV')
            html = catchup(expected=2, total=1)
            cache.save(1, URL, html)
            client = Mock(get=Mock(return_value=catchup()))
            papers, _ = crawl.category_catchup(client, 'cs.CV', DAY, root / 'cache', root / 'diagnostics')
            self.assertEqual(len(papers), 1)
            saved = list((root / 'diagnostics').rglob('diagnostic.json'))
            self.assertEqual(len(saved), 1)
            self.assertTrue(json.loads(saved[0].read_text())['pages'][0]['cached'])
            self.assertEqual(saved[0].with_name('page-1.html').read_text(), html)

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
            # The next daily run must resume, even several days after failure.
            with patch.object(crawl.time, 'time', return_value=crawl.time.time() + 3 * 86400):
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

    def test_cache_expiry_is_bounded_to_recovery_retention(self, _):
        with tempfile.TemporaryDirectory() as temp:
            cache = crawl.PageCache(Path(temp), DAY, 'cs.CV')
            cache.save(1, URL, catchup())
            now = crawl.time.time()
            with patch.object(crawl.time, 'time', return_value=now + 13 * 86400):
                self.assertIsNotNone(cache.load(1, URL))
            with patch.object(crawl.time, 'time', return_value=now + 15 * 86400):
                self.assertIsNone(cache.load(1, URL))

    def test_recovery_prefers_newest_page_over_existing_or_older_bundle(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, restored = root / 'data', root / 'restored'
            cache = crawl.PageCache(data / 'crawl_cache', DAY, 'cs.CV')
            archives = []
            for index, retrieved_at in enumerate((1000, 2000)):
                with patch.object(crawl.time, 'time', return_value=retrieved_at):
                    cache.save(1, URL, catchup(summary=f'version {index}'))
                bundle = root / f'bundle-{index}'
                save_bundle(data, root / 'stage', bundle, DAY)
                archive = root / f'bundle-{index}.zip'
                with zipfile.ZipFile(archive, 'w') as stream:
                    for path in bundle.rglob('*'):
                        if path.is_file():
                            stream.write(path, str(path.relative_to(bundle)))
                archives.append(archive)
            for archive in (archives[0], archives[1], archives[0]):
                restore_zip(archive, restored)
            page = json.loads((restored / f'crawl_cache/{DAY}/cs.CV/page-1.json').read_text())
            self.assertEqual(page['retrieved_at'], 2000)
            self.assertIn('version 1', page['html'])

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
            client = Mock(get=Mock(side_effect=[catchup(), catchup(summary='Changed abstract'), '<html></html>']))
            with patch.object(crawl, 'recall_categories', return_value=['cs.AI', 'cs.CV']):
                with self.assertRaisesRegex(ValueError, 'Metadata changed across category'):
                    crawl.crawl(DAY, root, root / 'run_metrics', client)
            self.assertEqual(previous.read_text(), 'previous successful output')
            self.assertFalse((root / f'run_metrics/{DAY}-crawl.json').exists())
            self.assertFalse((root / 'crawl_cache' / DAY).exists())


if __name__ == '__main__':
    unittest.main()
