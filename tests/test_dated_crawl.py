import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import crawl_dated as crawl


def listing(day="Wed, 30 Sep 2026", ids=("2609.10001",), expected=None, total=None, prefix=""):
    expected = len(ids) if expected is None else expected
    total = len(ids) if total is None else total
    entries = ''.join(f'<dt><a title="Abstract" href="/abs/{pid}">paper</a></dt><dd>Title</dd>' for pid in ids)
    return f'<div class="paging">Total of {total} entries</div><dl><h3>{day} (showing {prefix}{len(ids)} of {expected} entries )</h3>{entries}</dl>'


def atom(ids=("2609.10001",), summary="Complete abstract including $x^2$."):
    entries = ''.join(f'<entry><id>http://arxiv.org/abs/{pid}v2</id><title>A video world model</title><summary>{summary}</summary><author><name>A. Author</name></author><category term="cs.CV"/><published>2026-09-28T12:00:00Z</published><updated>2026-10-01T12:00:00Z</updated></entry>' for pid in ids)
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


class DatedCrawlTests(unittest.TestCase):
    def test_pagination_and_announcement_date(self):
        client = Mock()
        client.get.side_effect = [listing(expected=2, total=2, prefix="first "), listing(ids=("2609.10002",), expected=2, total=2, prefix="last ").replace("(showing", "(continued, showing")]
        days, urls = crawl.category_days(client, "cs.CV")
        self.assertEqual(days["2026-09-30"]["ids"], ["2609.10001", "2609.10002"])
        self.assertIn("skip=1", urls[1])

    def test_incomplete_or_changing_pagination_fails(self):
        for second in [listing(ids=("2609.10002",), expected=3, total=3), listing(expected=2, total=2)]:
            client = Mock()
            client.get.side_effect = [listing(expected=2, total=2), second]
            with self.assertRaises(ValueError):
                crawl.category_days(client, "cs.CV")
        with self.assertRaises(ValueError):
            crawl.parse_listing(listing().replace('showing 1', 'showing 2'))
        with self.assertRaises(ValueError):
            crawl.parse_listing('<dl><dt>today only</dt></dl>')

    def test_full_metadata_and_date_basis(self):
        result = crawl.parse_metadata(atom(), ["2609.10001"], "2026-09-30", {"2609.10001": ["cs.CV", "cs.AI"]})["2609.10001"]
        self.assertIn('$x^2$', result["summary"])
        self.assertEqual(result["announcement_date"], "2026-09-30")
        self.assertEqual(result["published"], "2026-09-28T12:00:00Z")
        self.assertEqual(result["metadata_version"], "2609.10001v2")

    def test_missing_or_duplicate_metadata_fails(self):
        for xml in [atom(ids=()), atom(ids=("2609.10001", "2609.10001")), atom(summary=""), atom(ids=("2609.10002",))]:
            with self.assertRaises(ValueError):
                crawl.parse_metadata(xml, ["2609.10001"], "2026-09-30", {"2609.10001": ["cs.CV"]})

    @patch.object(crawl, "recall_categories", return_value=["cs.CV", "cs.AI"])
    def test_cross_category_dedup_and_verified_snapshot(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock()
            client.get.side_effect = [listing(), listing(), atom()]
            audit = crawl.crawl("2026-09-30", root, root / "run_metrics", client)
            self.assertEqual(audit["unique_papers"], 1)
            self.assertEqual(client.get.call_count, 3)
            client.reset_mock()
            self.assertEqual(crawl.crawl("2026-09-30", root, root / "run_metrics", client), audit)
            client.get.assert_not_called()
            self.assertIsNone(crawl.verified_snapshot(root, "2026-09-30", ["cs.CV"]))
            (root / "2026-09-30.jsonl").write_text('{}\n')
            self.assertIsNone(crawl.verified_snapshot(root, "2026-09-30", ["cs.AI", "cs.CV"]))

    @patch.object(crawl, "recall_categories", return_value=["cs.CV"])
    def test_unknown_date_preserves_existing_output(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / "2026-09-29.jsonl"
            previous.write_text("old good output\n")
            client = Mock()
            client.get.return_value = listing()
            with self.assertRaisesRegex(ValueError, 'not verifiable'):
                crawl.crawl("2026-09-29", root, root / "audit", client)
            self.assertEqual(previous.read_text(), "old good output\n")
            self.assertFalse((root / "audit").exists())
            for day in ["2026-09-31", "2026-9-30", "2999-09-30", "../../invalid"]:
                with self.assertRaises(ValueError):
                    crawl.crawl(day, root, root / "audit", client)

    @patch.object(crawl, "recall_categories", return_value=["cs.CV"])
    def test_default_uses_latest_actual_announcement(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock()
            client.get.side_effect = [listing(), atom()]
            audit = crawl.crawl("", root, root / "audit", client)
            self.assertEqual(audit["target_date"], "2026-09-30")


if __name__ == "__main__":
    unittest.main()
