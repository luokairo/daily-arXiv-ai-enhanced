#!/usr/bin/env python3
"""Compare two GLM reasoning levels on one real paper without publishing data."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai"))
os.chdir(ROOT / "ai")  # enhance.py reads its prompt files relative to this directory.

import enhance  # noqa: E402
from structure import Structure  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    return parser.parse_args()


def find_paper(path: Path, paper_id: str):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                paper = json.loads(line)
                if paper.get("id") == paper_id:
                    return paper
    raise ValueError(f"Paper {paper_id} not found in {path.name}")


def print_result(effort: str, elapsed: float, response):
    raw = response.get("raw")
    parsed = response.get("parsed")
    usage = (getattr(raw, "response_metadata", {}) or {}).get("token_usage", {})
    print(f"\n=== reasoning_effort={effort} ===")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Usage: {json.dumps(usage, ensure_ascii=False)}")
    if parsed is None:
        print(f"Structured output failed: {response.get('parsing_error')}")
        return False
    result = parsed.model_dump()
    for key in (
        "is_relevant",
        "primary_direction_id",
        "subtopic_name",
        "tldr",
        "motivation",
        "method",
        "result",
        "conclusion",
    ):
        value = str(result.get(key, "")).replace("\n", " ")
        print(f"{key}: {value[:600]}")
    return True


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
    paper = find_paper(args.data.resolve(), args.paper_id)
    directions = enhance.load_directions(ROOT / "config" / "directions.yaml")
    taxonomy = enhance.load_taxonomy(ROOT / "data" / "taxonomy.json", directions)
    variables = {
        "language": "Chinese",
        "directions": enhance.compact_directions_for_prompt(directions),
        "taxonomy": enhance.compact_taxonomy_for_prompt(taxonomy),
        "title": paper.get("title", ""),
        "authors": ", ".join(paper.get("authors", [])),
        "categories": ", ".join(paper.get("categories", [])),
        "content": paper.get("summary", ""),
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(enhance.system),
            HumanMessagePromptTemplate.from_template(enhance.template),
        ]
    )
    print(f"Paper: {paper['id']} {paper.get('title', '')}")
    print("Two requests only; no paper data will be published.")
    successful = 0
    for effort in ("low", "max"):
        model = ChatOpenAI(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=6144,
            timeout=180,
            max_retries=0,
            reasoning_effort=effort,
        ).with_structured_output(Structure, method="function_calling", include_raw=True)
        start = time.monotonic()
        try:
            response = (prompt | model).invoke(variables)
            successful += print_result(effort, time.monotonic() - start, response)
        except Exception as exc:
            print(f"\n=== reasoning_effort={effort} ===")
            safe_error = str(exc).replace(api_key, "***REDACTED***")
            print(f"Request failed after {time.monotonic() - start:.1f}s: {type(exc).__name__}: {safe_error}")
    return 0 if successful == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
