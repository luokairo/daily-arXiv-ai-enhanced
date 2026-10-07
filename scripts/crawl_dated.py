"""Crawl an exact arXiv announcement day, never relabel /new as a past day.

The official catchup pages provide dated membership and full abstracts for the
past 90 days. Older dates require a previously verified snapshot.
Metadata/full texts are the currently available versions, not an as-of archive.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from parsel import Selector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_arxiv import load_directions, recall_categories

HEADER = re.compile(r"([A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4})\s*\((?:continued, )?showing (?:(?:first|last) )?(\d+) of (\d+) entr(?:y|ies)\s*\)")
SECTION = re.compile(r"(New|Cross|Replacement) submissions \(\s*(?:continued, )?showing (?:(?:first|last) )?(\d+) of (\d+) entr(?:y|ies)\s*\)")


def text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def paper_id(value):
    value = re.sub(r"v\d+$", "", value.rsplit("/abs/", 1)[-1])
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})", value):
        raise ValueError(f"Invalid arXiv ID: {value}")
    return value


def parse_listing(html):
    page = Selector(text=html)
    total_match = re.search(r"Total of\s+([\d,]+)\s+entr", text(" ".join(page.css(".paging").xpath("string()").getall())))
    if not total_match:
        raise ValueError("Cannot verify listing total")
    groups = {}
    for group in page.css("dl"):
        heading = text(group.css("h3").xpath("string()").get())
        match = HEADER.fullmatch(heading)
        if not match:
            raise ValueError(f"Cannot verify announcement heading: {heading!r}")
        day = datetime.strptime(match[1], "%a, %d %b %Y").date().isoformat()
        ids = [paper_id(link) for link in group.css('dt a[title="Abstract"]::attr(href)').getall()]
        if len(ids) != int(match[2]) or len(set(ids)) != len(ids):
            raise ValueError(f"Incomplete/duplicate entries for {day}")
        if day in groups:
            raise ValueError(f"Repeated announcement heading: {day}")
        groups[day] = {"expected": int(match[3]), "ids": ids}
    count = sum(len(group["ids"]) for group in groups.values())
    if not groups or count != len(page.css('dl dt')):
        raise ValueError("Unaccounted listing entries")
    return int(total_match[1].replace(",", "")), groups


class Client:
    def __init__(self, max_attempts=5, retry_budget=900):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "daily-arXiv-ai-enhanced/dated-crawl (https://github.com/luokairo/daily-arXiv-ai-enhanced)"
        self.last_request = 0.0
        self.max_attempts = max_attempts
        self.retry_budget = retry_budget
        self.current_url = None

    def get(self, url, **kwargs):
        started = time.monotonic()
        self.current_url = url
        for attempt in range(self.max_attempts):
            time.sleep(max(0, 15.1 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            print(f"Fetching {url} (attempt {attempt + 1}/{self.max_attempts})", flush=True)
            try:
                response = self.session.get(url, timeout=(20, 90), **kwargs)
                response.raise_for_status()
                return response.text
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                transient = status is None or status == 429 or 500 <= status < 600
                delay = max(60 * 2 ** attempt + random.uniform(0, 10),
                            retry_after_seconds(getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After", "")))
                remaining = self.retry_budget - (time.monotonic() - started)
                # Never shorten the server's cooldown to fit our budget. Leave
                # room for the next connection/read timeout, then fail cleanly.
                if not transient or attempt + 1 == self.max_attempts or delay + 110 > remaining:
                    print(f"arXiv fetch failed: {url}; {status or type(exc).__name__}; "
                          f"attempts={attempt + 1}, elapsed={time.monotonic() - started:.0f}s; "
                          "no further retries within policy/budget", flush=True)
                    raise
                print(f"Transient arXiv failure ({status or type(exc).__name__}) at {url}; "
                      f"retry {attempt + 1}/{self.max_attempts - 1} in {delay:.1f}s", flush=True)
                time.sleep(delay)
        raise RuntimeError("Unreachable")


def retry_after_seconds(value):
    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        deadline = parsedate_to_datetime(value)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0, deadline.timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        return 0


class PageCache:
    """Short-lived official HTML, revalidated before use; never a report."""
    def __init__(self, directory, day, category):
        self.directory = directory / day / category
        self.day, self.category, self.hits = day, category, 0

    def load(self, number, url):
        path = self.directory / f"page-{number}.json"
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entry, dict) or not isinstance(entry.get("html"), str):
                raise ValueError("Invalid cache schema")
            html = entry["html"]
            age = time.time() - entry["retrieved_at"]
            if (entry["schema_version"] != 1 or entry["day"] != self.day or
                    entry["category"] != self.category or entry["url"] != url or
                    entry["page"] != number or not 0 <= age <= 6 * 3600 or
                    hashlib.sha256(html.encode()).hexdigest() != entry["sha256"]):
                raise ValueError("Cache identity, checksum or freshness mismatch")
            parsed = parse_catchup(html, self.day, self.category)
        except (ValueError, KeyError, TypeError, OSError):
            print(f"Ignoring invalid/expired crawl page cache: {url}", flush=True)
            return None
        self.hits += 1
        print(f"Reusing verified crawl page: {url}", flush=True)
        return parsed

    def save(self, number, url, html):
        entry = dict(schema_version=1, day=self.day, category=self.category,
                     page=number, url=url, retrieved_at=time.time(), html=html,
                     sha256=hashlib.sha256(html.encode()).hexdigest())
        atomic_write(self.directory / f"page-{number}.json", json.dumps(entry, ensure_ascii=False) + "\n")

    def invalidate(self):
        shutil.rmtree(self.directory, ignore_errors=True)


def parse_catchup(html, day, category):
    page = Selector(text=html)
    expected_date = date.fromisoformat(day).strftime("%a, %d %b %Y")
    heading = text(page.css("#dlpage h1").xpath("string()").get())
    if not heading.startswith("Catchup results for ") or not heading.endswith(" on " + expected_date):
        raise ValueError(f"Catchup date mismatch: {heading!r}")
    totals = re.findall(r"Total of ([\d,]+) entries for " + re.escape(expected_date), text(page.css("#dlpage").xpath("string()").get()))
    if not totals or len(set(totals)) != 1:
        raise ValueError("Cannot verify catchup total")
    total, groups, papers = int(totals[0].replace(",", "")), {}, {}
    for group in page.css("dl"):
        match = SECTION.fullmatch(text(group.css("h3").xpath("string()").get()))
        if not match:
            raise ValueError("Unknown catchup section")
        kind, shown, expected = match[1], int(match[2]), int(match[3])
        ids = []
        for item in group.css("dt"):
            pid = paper_id(item.css('a[title="Abstract"]::attr(href)').get() or "")
            ids.append(pid)
            if kind == "Replacement":
                continue  # Preserve the original new/cross-list scope, no replacement re-analysis.
            dd = item.xpath("following-sibling::dd[1]")
            title = re.sub(r"^Title:\s*", "", text(dd.css(".list-title").xpath("string()").get()))
            abstract = text(dd.css("p.mathjax").xpath("string()").get())
            categories = re.findall(r"\(([^)]+)\)", text(dd.css(".list-subjects").xpath("string()").get()))
            if not title or not abstract or category not in categories or pid in papers:
                raise ValueError(f"Incomplete or duplicate catchup metadata: {pid}")
            papers[pid] = dict(id=pid, title=title, summary=abstract, categories=sorted(set(categories)),
                authors=[text(author) for author in dd.css(".list-authors a").xpath("string()").getall()],
                comment=re.sub(r"^Comments:\s*", "", text(dd.css(".list-comments").xpath("string()").get())),
                abs=f"https://arxiv.org/abs/{pid}", pdf=f"https://arxiv.org/pdf/{pid}",
                announcement_date=day, metadata_basis="current_available_version",
                announcement_categories=[category])
        if shown != len(ids) or len(ids) != len(set(ids)) or kind in groups:
            raise ValueError("Catchup section count mismatch")
        groups[kind] = dict(expected=expected, ids=ids)
    if sum(len(group['ids']) for group in groups.values()) != len(page.css("dl dt")):
        raise ValueError("Unaccounted catchup entries")
    if not groups and total != 0:
        raise ValueError("Missing catchup sections")
    return total, groups, papers


def category_catchup(client, category, day, cache_directory=None):
    cache = PageCache(cache_directory, day, category) if cache_directory else None
    try:
        return _category_catchup(client, category, day, cache)
    except ValueError:
        had_cached_pages = cache is not None and cache.hits > 0
        if cache:
            cache.invalidate()
        if not had_cached_pages:
            raise
        print(f"Cached pagination changed for {day} {category}; fetching category afresh", flush=True)
        # One fresh pass only; date/count errors must never become a partial day.
        return _category_catchup(client, category, day, cache)


def _category_catchup(client, category, day, cache):
    merged, papers, urls, page_number, count, total = {}, {}, [], 1, 0, None
    while total is None or count < total:
        url = f"https://arxiv.org/catchup/{category}/{day}?abs=True&page={page_number}"
        parsed = cache.load(page_number, url) if cache else None
        html = None
        if parsed is None:
            html = client.get(url)
            parsed = parse_catchup(html, day, category)
        current_total, groups, page_papers = parsed
        if total is not None and total != current_total:
            raise ValueError("Catchup changed during pagination")
        total = current_total
        urls.append(url)
        page_count = 0
        for kind, group in groups.items():
            existing = merged.setdefault(kind, dict(expected=group['expected'], ids=[]))
            if existing['expected'] != group['expected'] or set(existing['ids']) & set(group['ids']):
                raise ValueError("Inconsistent catchup pagination")
            existing['ids'].extend(group['ids'])
            page_count += len(group['ids'])
        if set(papers) & set(page_papers) or (not page_count and total):
            raise ValueError("Repeated or empty catchup page")
        if cache and html is not None:
            cache.save(page_number, url, html)
        papers.update(page_papers)
        count += page_count
        page_number += 1
    if count != total or sum(g['expected'] for g in merged.values()) != total or any(len(g['ids']) != g['expected'] for g in merged.values()):
        raise ValueError("Incomplete catchup day")
    return papers, dict(count=len(papers), listing_urls=urls,
                        replacements_excluded=len(merged.get('Replacement', {}).get('ids', [])))


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def verified_snapshot(directory, day, categories):
    raw = directory / f"{day}.jsonl"
    manifest = directory / "run_metrics" / f"{day}-crawl.json"
    if not raw.exists() or not manifest.exists():
        return None
    try:
        return _verified_snapshot(raw, manifest, day, categories)
    except (ValueError, KeyError, TypeError, OSError):
        return None


def _verified_snapshot(raw, manifest, day, categories):
    audit = json.loads(manifest.read_text())
    if not isinstance(audit, dict):
        return None
    if audit.get("status") != "complete" or audit.get("date_basis") != "arxiv_announcement_listing" or audit.get("target_date") != day or audit.get("categories") != sorted(categories):
        return None
    content = raw.read_text()
    if hashlib.sha256(content.encode()).hexdigest() != audit.get("raw_sha256"):
        return None
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        return None
    if len(rows) != audit.get("unique_papers") or len({row["id"] for row in rows}) != len(rows) or any(row.get("announcement_date") != day or not row.get("summary") for row in rows):
        return None
    return content, audit


def crawl(day, directory, audit_directory, client=None):
    if day and (not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or date.fromisoformat(day) > datetime.now(timezone.utc).date()):
        raise ValueError("Target must be a valid non-future YYYY-MM-DD announcement date")
    categories = sorted(set(recall_categories(load_directions(os.getenv("DIRECTIONS_CONFIG")))))
    if not categories:
        raise ValueError("No configured categories")
    client = client or Client()
    counts = {}
    cache_directory = directory / "crawl_cache"
    try:
        if not day:
            # Find the latest actual announcement day, including on weekends.
            _, groups = parse_listing(client.get(f"https://arxiv.org/list/{categories[0]}/pastweek?skip=0&show=25"))
            day = max(groups)
        print(f"Selected announcement date: {day}; categories: {', '.join(categories)}", flush=True)
        # Date selection enables recovery on failure; paper_count remains unset
        # until all categories have been verified. This is not a success signal.
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"crawl_date={day}\n")
        cached = verified_snapshot(directory, day, categories)
        if not cached:
            papers = {}
            for category in categories:
                category_papers, counts[category] = category_catchup(client, category, day, cache_directory)
                for pid, paper in category_papers.items():
                    if pid in papers:
                        if papers[pid]['title'] != paper['title'] or papers[pid]['summary'] != paper['summary']:
                            raise ValueError(f"Metadata changed across category snapshots: {pid}")
                        papers[pid]['announcement_categories'].append(category)
                    else:
                        papers[pid] = paper
                print(f"Verified {day} {category}: {len(category_papers)} full abstracts", flush=True)
    except Exception as exc:
        if isinstance(exc, ValueError) and day:
            shutil.rmtree(cache_directory / day, ignore_errors=True)
        progress = dict(schema_version=1, status="failed", target_date=day or None,
                        categories=categories, completed_categories=counts,
                        failed_url=getattr(client, "current_url", None) if isinstance(client, Client) else None,
                        error_type=type(exc).__name__, error=str(exc),
                        recorded_at=datetime.now(timezone.utc).isoformat())
        atomic_write(audit_directory / f"{day or 'latest'}-crawl-progress.json",
                     json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
        raise
    if cached:
        content, audit = cached
        print(f"Reusing verified announcement snapshot for {day}", flush=True)
    else:
        ids = sorted(papers)
        print(f"Announcement {day}: {len(ids)} unique papers", flush=True)
        content = "".join(json.dumps(papers[pid], ensure_ascii=False) + "\n" for pid in ids)
        audit = dict(schema_version=1, status="complete", target_date=day,
            date_basis="arxiv_announcement_listing", metadata_basis="current_available_version",
            retrieved_at=datetime.now(timezone.utc).isoformat(), categories=categories,
            category_counts=counts, unique_papers=len(ids),
            raw_sha256=hashlib.sha256(content.encode()).hexdigest())
    atomic_write(directory / f"{day}.jsonl", content)
    atomic_write(audit_directory / f"{day}-crawl.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"paper_count={audit['unique_papers']}\n")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(f"\n### Date-verified crawl\nAnnouncement date: **{day}**; {audit['unique_papers']} unique full abstracts. Metadata: current available versions.\n")
            for category, details in audit["category_counts"].items():
                summary.write(f"- {category}: {details['count']}\n")
    print(json.dumps({"date": day, "papers": audit["unique_papers"], "date_basis": audit["date_basis"]}), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="", help="Exact announcement date; omitted selects latest listed day")
    parser.add_argument("--directory", type=Path, default=Path("data"))
    parser.add_argument("--audit-directory", type=Path)
    args = parser.parse_args()
    crawl(args.date.strip(), args.directory, args.audit_directory or args.directory / "run_metrics")
