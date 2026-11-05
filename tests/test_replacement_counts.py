import unittest
import json
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

from scripts import crawl_dated as crawl
from scripts.recovery import restore_diagnostic_zip
from test_dated_crawl import catchup


def page(parts, total, next_page=False):
    first = catchup(total=total)
    header = first[:first.index('<dl>')]
    body = ''.join(crawl.Selector(text=part).css('dl').get() for part in parts)
    link = '<a href="/catchup/cs.CV/2026-09-30?abs=True&page=2">Next</a>' if next_page else ''
    return header + body + link + '</div>'


class ReplacementCountTests(unittest.TestCase):
    def run_page(self, html):
        return crawl.category_catchup(Mock(get=Mock(return_value=html)), 'cs.CV', '2026-09-30')

    def test_extra_replacement_is_warning_and_never_included_in_papers(self):
        html = page([catchup(), catchup(kind='Cross', ids=('2609.10002',)),
                     catchup(kind='Replacement', ids=('2601.10001', '2601.10002'), expected=1)], 3)
        papers, audit = self.run_page(html)
        self.assertEqual(set(papers), {'2609.10001', '2609.10002'})
        self.assertEqual(audit['replacements_excluded'], 2)
        self.assertEqual(audit['warnings'][0]['expected'], 1)
        self.assertEqual(audit['warnings'][0]['actual'], 2)

    def test_missing_replacement_without_next_page_is_warning(self):
        html = page([catchup(), catchup(kind='Replacement', ids=('2601.10001',), expected=2)], 3)
        client = Mock(get=Mock(return_value=html))
        papers, audit = crawl.category_catchup(client, 'cs.CV', '2026-09-30')
        self.assertEqual(len(papers), 1)
        self.assertEqual(len(audit['warnings']), 1)
        client.get.assert_called_once()

    def test_replacement_shown_count_mismatch_is_warning(self):
        replacement = catchup(kind='Replacement', ids=('2601.10001',)).replace('showing 1 of 1', 'showing 2 of 1')
        _, audit = self.run_page(page([catchup(), replacement], 2))
        self.assertEqual(audit['warnings'][0]['shown'], 2)
        self.assertEqual(audit['warnings'][0]['actual'], 1)

    def test_new_and_cross_must_be_complete_despite_replacement_surplus(self):
        for kind in ('New', 'Cross'):
            with self.subTest(kind=kind):
                html = page([catchup(kind=kind, expected=2),
                             catchup(kind='Replacement', ids=('2601.10001', '2601.10002'), expected=1)], 3)
                with self.assertRaises(crawl.CountMismatchError):
                    self.run_page(html)
                html = page([catchup(kind=kind, ids=('2609.10001', '2609.10002'), expected=1),
                             catchup(kind='Replacement', ids=('2601.10001',), expected=1)], 2)
                with self.assertRaises(crawl.CountMismatchError):
                    self.run_page(html)

    def test_unexplained_total_and_duplicate_ids_still_fail(self):
        with self.assertRaises(crawl.CountMismatchError):
            self.run_page(page([catchup(), catchup(kind='Replacement', ids=('2601.10001',))], 1))
        with self.assertRaises(ValueError):
            self.run_page(page([catchup(), catchup(kind='Replacement', ids=('2601.10001', '2601.10001'))], 3))

    def test_explicit_pagination_continues_when_replacements_exceed_total(self):
        first = page([catchup(), catchup(kind='Replacement', ids=('2601.10001', '2601.10002'), expected=1)], 3, True)
        second = page([catchup(kind='Cross', ids=('2609.10002',))], 3)
        client = Mock(get=Mock(side_effect=[first, second]))
        papers, audit = crawl.category_catchup(client, 'cs.CV', '2026-09-30')
        self.assertEqual(set(papers), {'2609.10001', '2609.10002'})
        self.assertEqual(client.get.call_count, 2)
        self.assertEqual(len(audit['warnings']), 1)


class DiagnosticRestoreTests(unittest.TestCase):
    def test_known_metadata_conflict_blocks_old_diagnostic_promotion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            html = page([catchup(), catchup(kind='Replacement', ids=('2601.10001', '2601.10002'), expected=1)], 2)
            archive, _ = self.archive(root, html)
            progress = root / 'data/run_metrics/2026-09-30-crawl-progress.json'
            progress.parent.mkdir(parents=True)
            progress.write_text(json.dumps(dict(target_date='2026-09-30',
                recorded_at=datetime.now(timezone.utc).isoformat(),
                error='Metadata changed across category snapshots: 2609.10001')))
            self.assertEqual(restore_diagnostic_zip(archive, root / 'data', '2026-09-30'), 0)
            self.assertFalse((root / 'data/crawl_cache').exists())

    def archive(self, root, html, age=2):
        retrieved = datetime.now(timezone.utc) - timedelta(days=age)
        url = 'https://arxiv.org/catchup/cs.CV/2026-09-30?abs=True&page=1'
        report = dict(target_date='2026-09-30', category='cs.CV', error_type='CountMismatchError',
                      recorded_at=retrieved.isoformat(), pages=[dict(page=1, url=url, file='page-1.html')])
        path = root / 'diagnostics.zip'
        with zipfile.ZipFile(path, 'w') as stream:
            stream.writestr('failure/diagnostic.json', json.dumps(report))
            stream.writestr('failure/page-1.html', html)
        return path, retrieved.timestamp()

    def test_verified_diagnostic_restores_original_age_and_does_not_replace_newer_cache(self):
        html = page([catchup(), catchup(kind='Replacement', ids=('2601.10001', '2601.10002'), expected=1)], 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, retrieved = self.archive(root, html)
            self.assertEqual(restore_diagnostic_zip(archive, root / 'data', '2026-09-30'), 1)
            cache = crawl.PageCache(root / 'data/crawl_cache', '2026-09-30', 'cs.CV')
            saved = cache.directory / 'page-1.json'
            self.assertEqual(json.loads(saved.read_text())['retrieved_at'], retrieved)
            client = Mock(get=Mock(side_effect=AssertionError('No network fetch should be needed')))
            papers, _ = crawl.category_catchup(client, 'cs.CV', '2026-09-30', root / 'data/crawl_cache')
            self.assertEqual(len(papers), 1)
            cache.save(1, 'https://arxiv.org/catchup/cs.CV/2026-09-30?abs=True&page=1', html)
            newer = saved.read_text()
            self.assertEqual(restore_diagnostic_zip(archive, root / 'data', '2026-09-30'), 0)
            self.assertEqual(saved.read_text(), newer)

    def test_incomplete_new_expired_and_wrong_date_diagnostics_are_never_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for html in (page([catchup(expected=2), catchup(kind='Replacement', ids=('2601.10001', '2601.10002'), expected=1)], 3),
                         catchup(expected=2, total=2)):
                archive, _ = self.archive(root, html)
                self.assertEqual(restore_diagnostic_zip(archive, root / 'data', '2026-09-30'), 0)
            html = page([catchup(), catchup(kind='Replacement', ids=('2601.10001', '2601.10002'), expected=1)], 2)
            archive, _ = self.archive(root, html, age=15)
            self.assertEqual(restore_diagnostic_zip(archive, root / 'data', '2026-09-30'), 0)
            archive, _ = self.archive(root, html)
            self.assertEqual(restore_diagnostic_zip(archive, root / 'data', '2026-09-29'), 0)
            self.assertFalse((root / 'data/crawl_cache').exists())


if __name__ == '__main__':
    unittest.main()
