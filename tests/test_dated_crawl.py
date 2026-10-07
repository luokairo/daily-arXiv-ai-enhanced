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


def catchup(day="2026-09-30", ids=("2609.10001",), kind="New", expected=None, total=None, summary="Complete <em>abstract</em> including $x^2$."):
    heading_date = crawl.date.fromisoformat(day).strftime("%a, %d %b %Y")
    expected = len(ids) if expected is None else expected
    total = len(ids) if total is None else total
    entries = ''.join(f'<dt><a title="Abstract" href="/abs/{pid}">paper</a></dt><dd><div class="list-title">Title: A video model</div><div class="list-authors"><a>A. Author</a></div><div class="list-subjects">Vision (cs.CV); AI (cs.AI)</div><p class="mathjax">{summary}</p></dd>' for pid in ids)
    return f'<div id="dlpage"><h1>Catchup results for Vision on {heading_date}</h1>Total of {total} entries for {heading_date}<dl><h3>{kind} submissions (showing {len(ids)} of {expected} entries)</h3>{entries}</dl></div>'


class DatedCrawlTests(unittest.TestCase):
    def test_latest_listing_date_and_partial_page(self):
        total, days = crawl.parse_listing(listing(expected=2, total=2, prefix="first "))
        self.assertEqual(total, 2)
        self.assertEqual(max(days), "2026-09-30")
        self.assertEqual(days["2026-09-30"]["expected"], 2)
        with self.assertRaises(ValueError):
            crawl.parse_listing(listing().replace('showing 1', 'showing 2'))
        with self.assertRaises(ValueError):
            crawl.parse_listing('<dl><dt>today only</dt></dl>')

    def test_full_metadata_and_date_basis(self):
        total, groups, papers = crawl.parse_catchup(catchup(), "2026-09-30", "cs.CV")
        result = papers["2609.10001"]
        self.assertEqual(result["summary"], "Complete abstract including $x^2$.")
        self.assertEqual(result["announcement_date"], "2026-09-30")
        self.assertEqual(result["metadata_basis"], "current_available_version")
        self.assertEqual(result["authors"], ["A. Author"])

    def test_missing_metadata_or_wrong_date_fails(self):
        for html in [catchup(summary=""), catchup(day="2026-09-29"), catchup(ids=("2609.10001", "2609.10001"))]:
            with self.assertRaises(ValueError):
                crawl.parse_catchup(html, "2026-09-30", "cs.CV")

    def test_catchup_pagination_and_replacement_exclusion(self):
        client = Mock()
        client.get.side_effect = [catchup(expected=2, total=3), catchup(ids=("2609.10002",), expected=2, total=3), catchup(ids=("2601.00001",), kind="Replacement", total=3)]
        papers, counts = crawl.category_catchup(client, "cs.CV", "2026-09-30")
        self.assertEqual(set(papers), {"2609.10001", "2609.10002"})
        self.assertEqual(counts['replacements_excluded'], 1)
        self.assertIn('page=3', counts['listing_urls'][2])
        client.get.side_effect = [catchup(expected=2, total=2), catchup(expected=2, total=2)]
        with self.assertRaises(ValueError):
            crawl.category_catchup(client, "cs.CV", "2026-09-30")

    @patch.object(crawl.time, "sleep")
    def test_http_failures_are_bounded(self, sleep):
        client = crawl.Client()
        response = crawl.requests.Response()
        response.status_code = 429
        response.headers['Retry-After'] = '45'
        client.session.get = Mock(return_value=response)
        with self.assertRaises(crawl.requests.HTTPError):
            client.get('https://arxiv.org/catchup/cs.CV/2026-09-30')
        self.assertEqual(client.session.get.call_count, 5)
        self.assertTrue(any(call.args[0] >= 60 for call in sleep.call_args_list))
        response.status_code = 404
        client.session.get.reset_mock()
        with self.assertRaises(crawl.requests.HTTPError):
            client.get('https://arxiv.org/catchup/cs.CV/2026-01-01')
        self.assertEqual(client.session.get.call_count, 1)

    @patch.object(crawl, "recall_categories", return_value=["cs.CV", "cs.AI"])
    def test_cross_category_dedup_and_verified_snapshot(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock()
            client.get.side_effect = [catchup(), catchup()]
            audit = crawl.crawl("2026-09-30", root, root / "run_metrics", client)
            self.assertEqual(audit["unique_papers"], 1)
            self.assertEqual(client.get.call_count, 2)
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
            client.get.return_value = catchup()
            with self.assertRaisesRegex(ValueError, 'date mismatch'):
                crawl.crawl("2026-09-29", root, root / "audit", client)
            self.assertEqual(previous.read_text(), "old good output\n")
            self.assertFalse((root / "audit/2026-09-29-crawl.json").exists())
            progress = json.loads((root / "audit/2026-09-29-crawl-progress.json").read_text())
            self.assertEqual(progress['status'], 'failed')
            for day in ["2026-09-31", "2026-9-30", "2999-09-30", "../../invalid"]:
                with self.assertRaises(ValueError):
                    crawl.crawl(day, root, root / "audit", client)

    @patch.object(crawl, "recall_categories", return_value=["cs.CV"])
    def test_default_uses_latest_actual_announcement(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock()
            client.get.side_effect = [listing(), catchup()]
            audit = crawl.crawl("", root, root / "audit", client)
            self.assertEqual(audit["target_date"], "2026-09-30")


if __name__ == "__main__":
    unittest.main()
