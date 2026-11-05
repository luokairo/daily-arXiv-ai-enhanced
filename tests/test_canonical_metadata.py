import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import crawl_dated as crawl
from test_dated_crawl import catchup


def abstract_page(summary='new', identity='2609.10001', title='A video model'):
    return (f'<meta name="citation_arxiv_id" content="{identity}">'
            f'<meta name="citation_title" content="{title}">'
            f'<h1 class="title">Title:{title}</h1>'
            f'<blockquote class="abstract">Abstract:{summary}</blockquote>')


class CanonicalMetadataTests(unittest.TestCase):
    def test_authoritative_page_selects_an_observed_version_and_preserves_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock(get=Mock(side_effect=[catchup(summary='old'), catchup(summary='new'), abstract_page()]))
            with patch.object(crawl, 'recall_categories', return_value=['cs.AI', 'cs.CV']):
                audit = crawl.crawl('2026-09-30', root, root / 'run_metrics', client)
            paper = json.loads((root / '2026-09-30.jsonl').read_text())
            self.assertEqual(paper['summary'], 'new')
            self.assertEqual(paper['announcement_categories'], ['cs.AI', 'cs.CV'])
            self.assertEqual(paper['metadata_source'], 'https://arxiv.org/abs/2609.10001')
            self.assertEqual(len(audit['metadata_reconciliations']), 1)
            evidence = root / 'crawl_diagnostics/2026-09-30/metadata/2609.10001.json'
            self.assertTrue(json.loads(evidence.read_text())['resolved'])
            self.assertTrue(evidence.with_suffix('.html').exists())
            client.get.assert_called_with('https://arxiv.org/abs/2609.10001')

    def test_wrong_identity_missing_data_and_neither_version_are_rejected(self):
        for html in (abstract_page(identity='2609.10002'), '<html></html>', abstract_page(summary='third version')):
            with self.subTest(html=html), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                client = Mock(get=Mock(side_effect=[catchup(summary='old'), catchup(summary='new'), html]))
                with patch.object(crawl, 'recall_categories', return_value=['cs.AI', 'cs.CV']):
                    with self.assertRaises(crawl.MetadataConflictError):
                        crawl.crawl('2026-09-30', root, root / 'run_metrics', client)
                self.assertFalse((root / '2026-09-30.jsonl').exists())

    def test_verified_page_is_requested_once_for_multiple_category_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pages = [catchup(summary='old'), catchup(summary='new'), abstract_page(), catchup(summary='old again')]
            pages = [html.replace('AI (cs.AI)', 'AI (cs.AI); Language (cs.CL)') for html in pages]
            client = Mock(get=Mock(side_effect=pages))
            with patch.object(crawl, 'recall_categories', return_value=['cs.AI', 'cs.CL', 'cs.CV']):
                audit = crawl.crawl('2026-09-30', root, root / 'run_metrics', client)
            paper = json.loads((root / '2026-09-30.jsonl').read_text())
            self.assertEqual(paper['summary'], 'new')
            self.assertEqual(paper['announcement_categories'], ['cs.AI', 'cs.CL', 'cs.CV'])
            self.assertEqual(len(audit['metadata_reconciliations']), 1)
            self.assertEqual(client.get.call_count, 4)


if __name__ == '__main__':
    unittest.main()
