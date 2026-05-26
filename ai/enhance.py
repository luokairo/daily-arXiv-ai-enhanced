import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import dotenv
import langchain_core.exceptions
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

from semantic_arxiv import (
    compact_directions_for_prompt,
    compact_taxonomy_for_prompt,
    direction_display,
    direction_map,
    load_directions,
    load_taxonomy,
    normalize_matched_directions,
    save_taxonomy,
    slugify,
    upsert_subtopic,
)
from structure import Structure

if os.path.exists(".env"):
    dotenv.load_dotenv()

template = open("template.txt", "r", encoding="utf-8").read()
system = open("system.txt", "r", encoding="utf-8").read()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument(
        "--directions",
        type=str,
        default=str(ROOT_DIR / "config" / "directions.yaml"),
        help="Path to semantic direction config",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=str(ROOT_DIR / "data" / "taxonomy.json"),
        help="Path to persistent subtopic taxonomy",
    )
    return parser.parse_args()


def default_ai_fields() -> Dict:
    return {
        "is_relevant": False,
        "primary_direction_id": "",
        "matched_direction_ids": [],
        "subtopic_id": "uncategorized",
        "subtopic_name": "未分类",
        "subtopic_description": "模型未能给出清晰定义的论文类别。",
        "is_new_subtopic": False,
        "classification_reason": "Classification unavailable",
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed",
    }


def is_sensitive(content: str) -> bool:
    if os.environ.get("ENABLE_SENSITIVE_CHECK", "").lower() not in {"1", "true", "yes"}:
        return False

    try:
        resp = requests.post(
            "https://spam.dw-dengwei.workers.dev",
            json={"text": content},
            timeout=5,
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("sensitive", False)
        print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Sensitive check error: {e}", file=sys.stderr)
        return False


def check_github_code(content: str) -> Dict:
    code_info = {}
    github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
    match = re.search(github_pattern, content)

    if match:
        owner, repo = match.groups()
        repo = repo.rstrip(".git").rstrip(".,)")
        code_info["code_url"] = f"https://github.com/{owner}/{repo}"

        github_token = os.environ.get("TOKEN_GITHUB")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                code_info["code_stars"] = data.get("stargazers_count", 0)
                code_info["code_last_update"] = data.get("pushed_at", "")[:10]
        except Exception:
            pass
        return code_info

    github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
    match_io = re.search(github_io_pattern, content)
    if match_io:
        code_info["code_url"] = match_io.group(0).rstrip(".,)")
    return code_info


def keyword_matched_direction_ids(item: Dict, directions: Iterable[Dict]) -> List[str]:
    text = " ".join(
        [
            item.get("title", ""),
            item.get("summary", ""),
            item.get("comment", "") or "",
            " ".join(item.get("categories", [])),
        ]
    ).lower()
    matched = []
    for direction in directions:
        keywords = [keyword.lower() for keyword in direction.get("keywords", [])]
        if any(keyword and keyword in text for keyword in keywords):
            matched.append(direction["id"])
    return matched


def fallback_item(item: Dict, directions: List[Dict], reason: str) -> Dict:
    matched_ids = keyword_matched_direction_ids(item, directions)
    if not matched_ids:
        return {}

    fields = default_ai_fields()
    fields.update(
        {
            "is_relevant": True,
            "primary_direction_id": matched_ids[0],
            "matched_direction_ids": matched_ids,
            "classification_reason": reason,
            "tldr": item.get("summary", "")[:600],
            "subtopic_id": "uncategorized",
            "subtopic_name": "未分类",
        }
    )
    item["AI"] = fields
    item["primary_direction"] = direction_display(matched_ids[0], directions)
    item["matched_directions"] = normalize_matched_directions(matched_ids, directions)
    return item


def prefilter_items(data: List[Dict], directions: List[Dict], max_items: int) -> List[Dict]:
    candidates = []
    for item in data:
        matched_ids = keyword_matched_direction_ids(item, directions)
        if not matched_ids:
            continue
        item["_prefilter_direction_ids"] = matched_ids
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            -len(item.get("_prefilter_direction_ids", [])),
            item.get("id", ""),
        )
    )

    if max_items > 0:
        candidates = candidates[:max_items]
    return candidates


def normalize_ai_result(item: Dict, ai_fields: Dict, directions: List[Dict]) -> Dict:
    valid_directions = direction_map(directions)
    ai = {**default_ai_fields(), **ai_fields}

    matched_ids = [direction_id for direction_id in ai.get("matched_direction_ids", []) if direction_id in valid_directions]
    primary_id = ai.get("primary_direction_id", "")
    if primary_id not in valid_directions:
        primary_id = matched_ids[0] if matched_ids else ""
    if primary_id and primary_id not in matched_ids:
        matched_ids.insert(0, primary_id)

    if not ai.get("is_relevant") or not primary_id:
        return {}

    subtopic_name = ai.get("subtopic_name") or "未分类"
    subtopic_id = ai.get("subtopic_id") or slugify(subtopic_name)
    if subtopic_id == "uncategorized" and subtopic_name != "未分类":
        subtopic_id = slugify(subtopic_name)
    if not re.match(r"^[a-z0-9][a-z0-9_-]{0,80}$", subtopic_id):
        subtopic_id = slugify(subtopic_name)

    ai["is_relevant"] = True
    ai["primary_direction_id"] = primary_id
    ai["matched_direction_ids"] = matched_ids
    ai["subtopic_id"] = subtopic_id
    ai["subtopic_name"] = subtopic_name
    ai["subtopic_description"] = ai.get("subtopic_description") or "模型未提供小方向定义。"

    item["AI"] = ai
    item["primary_direction"] = direction_display(primary_id, directions)
    item["matched_directions"] = normalize_matched_directions(matched_ids, directions)
    return item


def process_single_item(chain, item: Dict, language: str, directions: List[Dict], prompt_directions: str, prompt_taxonomy: str) -> Dict:
    if is_sensitive(item.get("summary", "")):
        return {}

    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    try:
        response: Structure = chain.invoke(
            {
                "language": language,
                "directions": prompt_directions,
                "taxonomy": prompt_taxonomy,
                "title": item.get("title", ""),
                "authors": ", ".join(item.get("authors", [])),
                "categories": ", ".join(item.get("categories", [])),
                "content": item.get("summary", ""),
            }
        )
        item = normalize_ai_result(item, response.model_dump(), directions)
    except langchain_core.exceptions.OutputParserException as e:
        print(f"Output parsing failed for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item = fallback_item(item, directions, "Structured output parsing failed; keyword fallback used.")
    except Exception as e:
        print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item = fallback_item(item, directions, "AI processing failed; keyword fallback used.")

    if not item:
        return {}

    for value in item.get("AI", {}).values():
        if is_sensitive(str(value)):
            return {}
    return item


def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int, directions: List[Dict], taxonomy: Dict) -> List[Dict]:
    llm_kwargs = {"model": model_name, "temperature": 0, "timeout": 120, "max_retries": 3}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url

    if model_name.startswith("deepseek-v4") and os.environ.get("DEEPSEEK_THINKING", "").lower() not in {"1", "true", "yes"}:
        llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    llm = ChatOpenAI(**llm_kwargs).with_structured_output(Structure, method="function_calling")
    print("Connect to:", model_name, file=sys.stderr)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(template=template),
        ]
    )
    chain = prompt_template | llm
    prompt_directions = compact_directions_for_prompt(directions)
    prompt_taxonomy = compact_taxonomy_for_prompt(taxonomy)

    processed_data = [{} for _ in data]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language, directions, prompt_directions, prompt_taxonomy): idx
            for idx, item in enumerate(data)
        }

        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Processing items"):
            idx = future_to_idx[future]
            try:
                processed_data[idx] = future.result()
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                processed_data[idx] = fallback_item(data[idx], directions, "Unhandled worker error; keyword fallback used.")

    return [item for item in processed_data if item]


def update_taxonomy_from_items(taxonomy: Dict, items: List[Dict], directions: List[Dict]) -> Dict:
    today = os.environ.get("TARGET_DATE") or datetime.utcnow().strftime("%Y-%m-%d")
    valid_directions = direction_map(directions)

    for item in items:
        ai = item.get("AI", {})
        direction_id = ai.get("primary_direction_id", "")
        if direction_id not in valid_directions:
            continue
        subtopic = upsert_subtopic(
            taxonomy,
            direction_id,
            ai.get("subtopic_id", ""),
            ai.get("subtopic_name", "未分类"),
            ai.get("subtopic_description", ""),
            item,
            now=today,
        )
        ai["subtopic_id"] = subtopic["id"]
        ai["subtopic_name"] = subtopic["name"]
        ai["subtopic_description"] = subtopic["description"]
        item["AI"] = ai

    return taxonomy


def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    language = os.environ.get("LANGUAGE", "Chinese")
    directions = load_directions(args.directions)
    if not directions:
        raise RuntimeError(f"No semantic directions found in {args.directions}")

    taxonomy = load_taxonomy(args.taxonomy, directions)

    target_file = args.data.replace(".jsonl", f"_AI_enhanced_{language}.jsonl")
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f"Removed existing file: {target_file}", file=sys.stderr)

    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_data.append(item)

    max_items = int(os.environ.get("MAX_AI_ITEMS", "120"))
    prefiltered_data = prefilter_items(unique_data, directions, max_items)
    print("Open:", args.data, file=sys.stderr)
    print(
        f"Prefiltered {len(prefiltered_data)} candidate papers from {len(unique_data)} crawled papers. "
        f"MAX_AI_ITEMS={max_items}",
        file=sys.stderr,
    )

    processed_data = process_all_items(prefiltered_data, model_name, language, args.max_workers, directions, taxonomy)
    taxonomy = update_taxonomy_from_items(taxonomy, processed_data, directions)
    save_taxonomy(taxonomy, args.taxonomy)

    with open(target_file, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Kept {len(processed_data)} semantically relevant papers", file=sys.stderr)


if __name__ == "__main__":
    main()
