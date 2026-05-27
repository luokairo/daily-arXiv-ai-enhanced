import argparse
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dotenv
import requests
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from semantic_arxiv import load_importance_config

if os.path.exists(".env"):
    dotenv.load_dotenv()


DEEP_READ_SYSTEM = """You are a careful bilingual research paper reader.
Analyze extracted arXiv PDF text using the user's paper-reader requirements:
motivation, core methodology, insight analysis, experimental design, limitations,
future work, and personal takeaways. Write in Chinese for fast reading, keep key
technical terms in English, and include concise English summaries where requested.
If the extracted text does not contain enough evidence for a field, say so explicitly.
Return Markdown only. Do not wrap the answer in code fences."""


DEEP_READ_TEMPLATE = """Read this paper for a daily research briefing.

Use exactly this structure for the report:

# {title}

**Authors**: {authors}
**Venue**: arXiv
**Link**: {abs_url}
**PDF**: {pdf_url}
**Importance**: {importance_level} ({importance_score})
**Why selected**: {importance_reason}
**Source note**: {source_note}

---

## 一句话总结 / One-liner

One sentence in Chinese, followed by one concise English sentence.

---

## 动机 / Motivation

### 中文概述
- What problem is the paper solving, and why does it matter?
- What gap in existing work does it target?
- What is the key observation or starting point?

### English Summary
Brief English summary of the motivation.

---

## 核心方法 / Core Method

### 中文概述
- Describe the overall framework or pipeline.
- Explain the central mechanism conceptually.
- Clarify how it differs from closest related work.

### English Summary
Brief English summary of the core method.

---

## 启发性分析 / Insight Analysis

### 中文概述
- What do ablations, qualitative analysis, or error analysis reveal?
- Any counterintuitive findings?
- Any scaling, generalization, or theory-related insight?
- If the paper lacks deep insight analysis, state that explicitly.

### English Summary
Brief English summary of key insights.

---

## 实验设计 / Experimental Design

| 维度 | 内容 |
|------|------|
| 数据集 | Main datasets and why they fit the claims |
| Baseline | Main baselines and whether they are fair |
| 评估指标 | Core metrics |
| 消融实验 | What was ablated and what was learned |
| 统计显著性 | Error bars, multiple runs, or significance tests if available |

---

## 局限性与展望 / Limitations & Future Work

### 中文概述
- Author-acknowledged limitations.
- Important unmentioned limitations you notice.
- Future directions.

### English Summary
Brief English summary of limitations and future work.

---

## 个人思考 / Personal Takeaways

- Critical assessment and useful ideas for follow-up research.

Paper metadata:
Title: {title}
Authors: {authors}
arXiv categories: {categories}
Primary direction: {direction}
Subtopic: {subtopic}
TL;DR: {tldr}
Abstract: {abstract}

Extracted PDF text:
{paper_text}
"""


SECTION_PATTERNS = {
    "abstract": [r"abstract"],
    "introduction": [r"introduction"],
    "related_work": [r"related work", r"background"],
    "method": [r"method", r"methods", r"methodology", r"approach", r"framework"],
    "experiments": [r"experiment", r"experiments", r"experimental setup", r"results", r"evaluation"],
    "limitations": [r"limitation", r"limitations", r"discussion"],
    "conclusion": [r"conclusion", r"conclusions", r"future work"],
}

HEADING_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(abstract|introduction|related work|background|method|methods|methodology|approach|framework|"
    r"experiment|experiments|experimental setup|results|evaluation|discussion|limitation|limitations|"
    r"conclusion|conclusions|future work|references|appendix)\b"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="AI enhanced JSONL file")
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument("--output-json", default="", help="Optional JSON summary output path")
    parser.add_argument(
        "--directions",
        default=str(ROOT_DIR / "config" / "directions.yaml"),
        help="Path to semantic direction config",
    )
    return parser.parse_args()


def safe_authors(authors) -> str:
    if isinstance(authors, list):
        return ", ".join(authors)
    return authors or ""


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_chat_model(model_name: str):
    llm_kwargs = {"model": model_name, "temperature": 0, "timeout": 240, "max_retries": 2}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url
    deep_read_thinking = os.environ.get("DEEP_READ_THINKING", "").lower() in {"1", "true", "yes"}
    if model_name.startswith("deepseek-v4") and not deep_read_thinking:
        llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**llm_kwargs)


def selected_papers(data: List[Dict], top_k: int) -> List[Dict]:
    selected = [
        item
        for item in data
        if isinstance(item.get("AI"), dict) and item["AI"].get("deep_read_selected")
    ]
    if not selected:
        selected = sorted(
            data,
            key=lambda item: float((item.get("AI") or {}).get("importance_score") or 0.0),
            reverse=True,
        )[:top_k]

    selected.sort(
        key=lambda item: (
            (item.get("AI") or {}).get("deep_read_rank") or 999,
            -float((item.get("AI") or {}).get("importance_score") or 0.0),
        )
    )
    return selected[:top_k]


def download_pdf(item: Dict, target_dir: Path) -> Path:
    pdf_url = item.get("pdf") or f"https://arxiv.org/pdf/{item.get('id', '')}"
    output_path = target_dir / f"{item.get('id', 'paper').replace('/', '_')}.pdf"
    headers = {"User-Agent": "daily-arxiv-ai-enhanced/1.0"}
    response = requests.get(pdf_url, headers=headers, timeout=90)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def extract_with_markitdown(pdf_path: Path) -> str:
    try:
        from markitdown import MarkItDown

        result = MarkItDown(enable_plugins=False).convert(pdf_path)
        return getattr(result, "markdown", "") or ""
    except Exception as e:
        print(f"MarkItDown extraction failed for {pdf_path.name}: {e}", file=sys.stderr)
        return ""


def extract_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    except Exception as e:
        print(f"pypdf extraction failed for {pdf_path.name}: {e}", file=sys.stderr)
        return ""


def extract_pdf_text(pdf_path: Path) -> str:
    text = extract_with_markitdown(pdf_path)
    if len(text.strip()) >= 2000:
        return text
    fallback = extract_with_pypdf(pdf_path)
    return fallback if len(fallback.strip()) > len(text.strip()) else text


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_section(text: str, patterns: List[str], max_chars: int = 12000) -> str:
    lower_patterns = [pattern.lower() for pattern in patterns]
    headings = list(HEADING_RE.finditer(text))
    for idx, heading in enumerate(headings):
        heading_text = heading.group(1).lower()
        if not any(pattern in heading_text for pattern in lower_patterns):
            continue
        start = heading.start()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else min(len(text), start + max_chars)
        return text[start:min(end, start + max_chars)].strip()
    return ""


def append_chunk(chunks: List[Tuple[str, str]], title: str, chunk: str, seen: set) -> None:
    chunk = normalize_text(chunk)
    if not chunk:
        return
    signature = chunk[:300]
    if signature in seen:
        return
    seen.add(signature)
    chunks.append((title, chunk))


def build_reading_context(item: Dict, full_text: str) -> Tuple[str, str]:
    full_text = normalize_text(full_text)
    max_chars = int(os.environ.get("DEEP_READ_MAX_CONTEXT_CHARS") or "70000")
    chunks: List[Tuple[str, str]] = []
    seen = set()

    append_chunk(chunks, "Metadata abstract", item.get("summary", ""), seen)
    append_chunk(chunks, "Front matter and early paper text", full_text[:16000], seen)

    for name, patterns in SECTION_PATTERNS.items():
        append_chunk(chunks, name, find_section(full_text, patterns), seen)

    context_parts = []
    total = 0
    for title, chunk in chunks:
        block = f"\n\n## {title}\n{chunk}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 2000:
                context_parts.append(block[:remaining])
            break
        context_parts.append(block)
        total += len(block)

    context = "\n".join(context_parts).strip()
    source_note = "PDF full text extracted; high-value sections were selected for the model context."
    if len(full_text) < 2000:
        context = item.get("summary", "")
        source_note = "PDF extraction produced too little text; this is an abstract-only fallback."
    return context, source_note


def failure_report(item: Dict, rank: int, reason: str) -> str:
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    return (
        f"## [{rank}] {item.get('title', 'Untitled')}\n\n"
        f"**Authors**: {safe_authors(item.get('authors'))}\n\n"
        f"**Link**: {item.get('abs', '')}\n\n"
        f"**PDF**: {item.get('pdf', '')}\n\n"
        f"**Importance**: {ai.get('importance_level', '')} ({ai.get('importance_score', '')})\n\n"
        f"**Why selected**: {ai.get('importance_reason', '')}\n\n"
        f"> Full-text deep read was not generated: {reason}\n"
    )


def run_deep_read(chain, item: Dict, rank: int, tmp_dir: Path) -> Dict:
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    try:
        pdf_path = download_pdf(item, tmp_dir)
        full_text = extract_pdf_text(pdf_path)
        paper_text, source_note = build_reading_context(item, full_text)
    except Exception as e:
        print(f"PDF download/extraction failed for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        paper_text = item.get("summary", "")
        source_note = f"PDF download or extraction failed; abstract-only fallback. Error: {e}"

    try:
        response = chain.invoke(
            {
                "title": item.get("title", "Untitled"),
                "authors": safe_authors(item.get("authors")),
                "abs_url": item.get("abs") or f"https://arxiv.org/abs/{item.get('id', '')}",
                "pdf_url": item.get("pdf") or f"https://arxiv.org/pdf/{item.get('id', '')}",
                "importance_level": ai.get("importance_level", ""),
                "importance_score": ai.get("importance_score", ""),
                "importance_reason": ai.get("importance_reason", ""),
                "source_note": source_note,
                "categories": ", ".join(item.get("categories", [])),
                "direction": ai.get("primary_direction_id", ""),
                "subtopic": ai.get("subtopic_name", ""),
                "tldr": ai.get("tldr", ""),
                "abstract": item.get("summary", ""),
                "paper_text": paper_text,
            }
        )
        content = response.content if hasattr(response, "content") else str(response)
        report = f"## [{rank}] 精读报告\n\n{content.strip()}\n"
        status = "ok" if "abstract-only fallback" not in source_note else "abstract_fallback"
    except Exception as e:
        print(f"Deep read generation failed for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        report = failure_report(item, rank, str(e))
        status = "generation_failed"

    return {
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "rank": rank,
        "status": status,
        "report": report,
    }


def output_date_from_path(path: str) -> str:
    name = Path(path).name
    match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return match.group(1) if match else "unknown-date"


def main():
    args = parse_args()
    data = load_jsonl(args.data)
    importance_config = load_importance_config(args.directions)
    top_k = int(os.environ.get("DAILY_DEEP_READ_TOP_K") or importance_config.get("daily_deep_read_top_k", 5) or 5)
    model_name = os.environ.get("DEEP_READ_MODEL_NAME") or os.environ.get("MODEL_NAME", "deepseek-v4-pro")
    max_workers = int(os.environ.get("DEEP_READ_MAX_WORKERS") or "1")

    papers = selected_papers(data, top_k)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_label = output_date_from_path(args.output)
    if not papers:
        output_path.write_text(
            f"# {date_label} 重要论文全文精读\n\nNo papers were selected for deep reading.\n",
            encoding="utf-8",
        )
        if args.output_json:
            Path(args.output_json).write_text("[]\n", encoding="utf-8")
        print(f"No papers selected. Wrote {output_path}", file=sys.stderr)
        return

    llm = build_chat_model(model_name)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(DEEP_READ_SYSTEM),
            HumanMessagePromptTemplate.from_template(DEEP_READ_TEMPLATE),
        ]
    )
    chain = prompt_template | llm

    results = [{} for _ in papers]
    with tempfile.TemporaryDirectory(prefix="daily_arxiv_deep_read_") as tmp:
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(run_deep_read, chain, item, idx + 1, tmp_dir): idx
                for idx, item in enumerate(papers)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(papers), desc="Deep reading papers"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {
                        "id": papers[idx].get("id", ""),
                        "title": papers[idx].get("title", ""),
                        "rank": idx + 1,
                        "status": "worker_failed",
                        "report": failure_report(papers[idx], idx + 1, str(e)),
                    }

    toc = "\n".join(
        f"- [{result.get('rank')}. {result.get('title')}](#paper-{result.get('rank')})"
        for result in results
    )
    reports = []
    for result in results:
        reports.append(f"<div id='paper-{result.get('rank')}'></div>\n\n{result.get('report', '')}")

    markdown = (
        f"# {date_label} 重要论文全文精读\n\n"
        f"Model: `{model_name}`\n\n"
        f"Selected papers: {len(results)}\n\n"
        "## Table of Contents\n\n"
        f"{toc}\n\n"
        "---\n\n"
        + "\n\n---\n\n".join(reports)
        + "\n"
    )
    output_path.write_text(markdown, encoding="utf-8")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {key: value for key, value in result.items() if key != "report"}
            for result in results
        ]
        json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
