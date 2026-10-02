"""Crawl an exact arXiv announcement day, never relabel /new as a past day.

The public pastweek lists provide daily membership; the Atom API supplies full
abstracts for those IDs. Older dates require a previously verified snapshot.
Metadata/full texts are the currently available versions, not an as-of archive.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from parsel import Selector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_arxiv import load_directions, recall_categories

HEADER = re.compile(r"([A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4})\s*\((?:continued, )?showing (?:(?:first|last) )?(\d+) of (\d+) entr(?:y|ies)\s*\)")
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


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
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "daily-arXiv-ai-enhanced/dated-crawl (https://github.com/luokairo/daily-arXiv-ai-enhanced)"
        self.last_request = 0.0

    def get(self, url, **kwargs):
        for attempt in range(3):
            time.sleep(max(0, 3.1 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.session.get(url, timeout=90, **kwargs)
                response.raise_for_status()
                return response.text
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 2 or (status is not None and status != 429 and status < 500):
                    raise
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After", "")
                delay = max(30 * (attempt + 1), int(retry_after) if retry_after.isdigit() else 0)
                print(f"Transient arXiv failure ({status or type(exc).__name__}); retry {attempt + 1}/2 in {delay}s", flush=True)
                time.sleep(delay)
        raise RuntimeError("Unreachable")


def category_days(client, category):
    merged, urls, offset, total = {}, [], 0, None
    while total is None or offset < total:
        url = f"https://arxiv.org/list/{category}/pastweek?skip={offset}&show=2000"
        current_total, groups = parse_listing(client.get(url))
        if total is not None and total != current_total:
            raise ValueError(f"Listing changed during pagination: {category}")
        total = current_total
        urls.append(url)
        count = 0
        for day, group in groups.items():
            existing = merged.setdefault(day, {"expected": group["expected"], "ids": []})
            if existing["expected"] != group["expected"] or set(existing["ids"]) & set(group["ids"]):
                raise ValueError(f"Inconsistent pagination: {category}/{day}")
            existing["ids"].extend(group["ids"])
            count += len(group["ids"])
        if not count:
            raise ValueError(f"Pagination did not advance: {category}")
        offset += count
    if offset != total:
        raise ValueError(f"Listing count mismatch: {category}")
    for day, group in merged.items():
        if len(group["ids"]) != group["expected"]:
            raise ValueError(f"Incomplete day: {category}/{day}")
    return merged, urls


def parse_metadata(xml, requested, day, sources):
    root = ET.fromstring(xml)
    papers = {}
    for entry in root.findall("a:entry", NS):
        identity = entry.findtext("a:id", "", NS)
        pid = paper_id(identity)
        title = text(entry.findtext("a:title", "", NS))
        abstract = text(entry.findtext("a:summary", "", NS))
        if pid not in requested or pid in papers or not title or not abstract:
            raise ValueError(f"Missing, duplicate or unexpected metadata: {pid}")
        papers[pid] = dict(id=pid, title=title, summary=abstract,
            authors=[text(node.text) for node in entry.findall("a:author/a:name", NS)],
            categories=[node.attrib["term"] for node in entry.findall("a:category", NS)],
            comment=text(entry.findtext("x:comment", "", NS)),
            abs=f"https://arxiv.org/abs/{pid}", pdf=f"https://arxiv.org/pdf/{pid}",
            published=entry.findtext("a:published", "", NS), updated=entry.findtext("a:updated", "", NS),
            metadata_version=identity.rsplit("/abs/", 1)[-1], announcement_date=day,
            announcement_categories=sorted(sources[pid]))
    if set(papers) != set(requested):
        raise ValueError(f"API omitted {len(set(requested) - set(papers))} requested papers")
    return papers


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
    audit = json.loads(manifest.read_text())
    if audit.get("status") != "complete" or audit.get("date_basis") != "arxiv_announcement_listing" or audit.get("target_date") != day or audit.get("categories") != sorted(categories):
        return None
    content = raw.read_text()
    if hashlib.sha256(content.encode()).hexdigest() != audit.get("raw_sha256"):
        return None
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
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
    cached = verified_snapshot(directory, day, categories) if day else None
    if cached:
        content, audit = cached
        print(f"Reusing verified announcement snapshot for {day}", flush=True)
    else:
        listings = {}
        for category in categories:
            listings[category] = category_days(client, category)
            print(f"Verified announcement listing: {category}", flush=True)
        available = set.union(*(set(groups) for groups, _ in listings.values()))
        day = day or max(available)
        # Missing dates are ambiguous (no announcements vs outside the public window).
        # Fail closed; never substitute today's list or a submittedDate query.
        sources, counts = {}, {}
        for category, (groups, urls) in listings.items():
            if day not in groups:
                raise ValueError(f"{day} is not verifiable in {category}/pastweek; use a verified saved snapshot. No output replaced.")
            ids = groups[day]["ids"]
            counts[category] = {"count": len(ids), "listing_urls": urls}
            for pid in ids:
                sources.setdefault(pid, []).append(category)
        ids = sorted(sources)
        print(f"Announcement {day}: {len(ids)} unique papers; " + json.dumps({cat: details['count'] for cat, details in counts.items()}), flush=True)
        papers = {}
        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            xml = client.get("https://export.arxiv.org/api/query", params={"id_list": ",".join(batch), "max_results": len(batch)})
            papers.update(parse_metadata(xml, batch, day, sources))
            print(f"Full abstracts: {len(papers)}/{len(ids)}", flush=True)
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
            output.write(f"crawl_date={day}\npaper_count={audit['unique_papers']}\n")
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
