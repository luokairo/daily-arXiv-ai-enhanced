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
    load_importance_config,
    load_taxonomy,
    normalize_matched_directions,
    save_taxonomy,
    slugify,
    upsert_subtopic,
)
from structure import FilterStructure, ImportanceStructure, Structure

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
        "importance_score": 0.0,
        "importance_level": "普通相关",
        "importance_reason": "Importance scoring unavailable",
        "deep_read_selected": False,
        "deep_read_rank": None,
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


def build_llm(model_name: str, structure_model):
    llm_kwargs = {"model": model_name, "temperature": 0, "timeout": 120, "max_retries": 3}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url

    if model_name.startswith("deepseek-v4") and os.environ.get("DEEPSEEK_THINKING", "").lower() not in {"1", "true", "yes"}:
        llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatOpenAI(**llm_kwargs).with_structured_output(structure_model, method="function_calling")


def normalize_filter_result(item: Dict, filter_fields: Dict, directions: List[Dict]) -> Dict:
    valid_directions = direction_map(directions)
    matched_ids = [direction_id for direction_id in filter_fields.get("matched_direction_ids", []) if direction_id in valid_directions]
    primary_id = filter_fields.get("primary_direction_id", "")
    if primary_id not in valid_directions:
        primary_id = matched_ids[0] if matched_ids else ""
    if primary_id and primary_id not in matched_ids:
        matched_ids.insert(0, primary_id)

    if not filter_fields.get("is_relevant") or not primary_id:
        return {}

    item["_filter_primary_direction_id"] = primary_id
    item["_filter_matched_direction_ids"] = matched_ids
    item["_filter_relevance_reason"] = filter_fields.get("relevance_reason", "")
    return item


def filter_single_item(chain, item: Dict, directions: List[Dict], prompt_directions: str) -> Dict:
    try:
        response: FilterStructure = chain.invoke(
            {
                "directions": prompt_directions,
                "title": item.get("title", ""),
                "authors": ", ".join(item.get("authors", [])),
                "categories": ", ".join(item.get("categories", [])),
                "content": item.get("summary", ""),
            }
        )
        item = normalize_filter_result(item, response.model_dump(), directions)
    except Exception as e:
        print(f"Filter error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item = fallback_item(item, directions, "Filter model failed; keyword fallback used.")
        if item:
            ai = item.pop("AI", {})
            item["_filter_primary_direction_id"] = ai.get("primary_direction_id", "")
            item["_filter_matched_direction_ids"] = ai.get("matched_direction_ids", [])
            item["_filter_relevance_reason"] = ai.get("classification_reason", "")
            item.pop("primary_direction", None)
            item.pop("matched_directions", None)
    return item


def filter_all_items(data: List[Dict], filter_model_name: str, max_workers: int, directions: List[Dict]) -> List[Dict]:
    filter_system = (
        "You are a fast research paper router. Decide if the paper belongs to one or more target research directions. "
        "Return only the structured fields. Do not summarize the paper."
    )
    filter_template = (
        "Target research directions:\n{directions}\n\n"
        "Paper metadata:\n"
        "Title: {title}\n"
        "Authors: {authors}\n"
        "arXiv categories: {categories}\n\n"
        "Abstract:\n{content}\n\n"
        "Task: classify relevance to the target directions. Use exact direction ids."
    )
    llm = build_llm(filter_model_name, FilterStructure)
    print("Filter model:", filter_model_name, file=sys.stderr)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(filter_system),
            HumanMessagePromptTemplate.from_template(template=filter_template),
        ]
    )
    chain = prompt_template | llm
    prompt_directions = compact_directions_for_prompt(directions)

    filtered_data = [{} for _ in data]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(filter_single_item, chain, item, directions, prompt_directions): idx
            for idx, item in enumerate(data)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Filtering items"):
            idx = future_to_idx[future]
            try:
                filtered_data[idx] = future.result()
            except Exception as e:
                print(f"Filter worker error at index {idx}: {e}", file=sys.stderr)

    return [item for item in filtered_data if item]


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
        if item and item.get("_filter_relevance_reason"):
            item["AI"]["filter_reason"] = item["_filter_relevance_reason"]
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
    llm = build_llm(model_name, Structure)
    print("Detail model:", model_name, file=sys.stderr)

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


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def item_text(item: Dict) -> str:
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    return " ".join(
        [
            item.get("title", ""),
            item.get("summary", ""),
            item.get("comment", "") or "",
            " ".join(item.get("categories", [])),
            ai.get("subtopic_name", ""),
            ai.get("classification_reason", ""),
            ai.get("tldr", ""),
        ]
    ).lower()


def heuristic_importance_score(item: Dict, importance_config: Dict) -> float:
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    text = item_text(item)
    score = 45.0

    direction_id = ai.get("primary_direction_id", "")
    direction_weight = importance_config.get("direction_weights", {}).get(direction_id, 1.0)
    score += (float(direction_weight) - 1.0) * 35.0

    for keyword in importance_config.get("boost_keywords", []):
        if keyword and keyword.lower() in text:
            score += 4.0

    for keyword in importance_config.get("negative_keywords", []):
        if keyword and keyword.lower() in text:
            score -= 10.0

    subtopic_name = ai.get("subtopic_name", "").lower()
    for subtopic in importance_config.get("priority_subtopics", []):
        name = str(subtopic.get("name", "")).lower()
        keywords = [str(keyword).lower() for keyword in subtopic.get("keywords", [])]
        if (name and name in subtopic_name) or any(keyword and keyword in text for keyword in keywords):
            score += 12.0 + (float(subtopic.get("weight", 1.2)) - 1.0) * 30.0

    if item.get("code_url"):
        score += 5.0
    if re.search(r"\b(first|novel|new|unified|state-of-the-art|sota|benchmark|dataset)\b", text):
        score += 5.0

    return clamp_score(score)


def importance_level(score: float) -> str:
    if score >= 80:
        return "高优先级"
    if score >= 65:
        return "值得关注"
    return "普通相关"


def compact_importance_config_for_prompt(importance_config: Dict) -> str:
    lines = [
        f"daily_deep_read_top_k: {importance_config.get('daily_deep_read_top_k', 5)}",
        "direction_weights:",
    ]
    for direction_id, weight in importance_config.get("direction_weights", {}).items():
        lines.append(f"  - {direction_id}: {weight}")
    lines.append("priority_subtopics:")
    for subtopic in importance_config.get("priority_subtopics", []):
        keywords = ", ".join(subtopic.get("keywords", []))
        lines.append(f"  - {subtopic.get('name')} (weight: {subtopic.get('weight', 1.2)}; keywords: {keywords})")
    lines.append("boost_keywords: " + ", ".join(importance_config.get("boost_keywords", [])))
    lines.append("negative_keywords: " + ", ".join(importance_config.get("negative_keywords", [])))
    return "\n".join(lines)


def score_single_importance(chain, item: Dict, importance_config_text: str, heuristic_score: float) -> Dict:
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    try:
        response: ImportanceStructure = chain.invoke(
            {
                "importance_config": importance_config_text,
                "title": item.get("title", ""),
                "authors": ", ".join(item.get("authors", [])),
                "categories": ", ".join(item.get("categories", [])),
                "direction": ai.get("primary_direction_id", ""),
                "subtopic": ai.get("subtopic_name", ""),
                "tldr": ai.get("tldr", ""),
                "abstract": item.get("summary", ""),
            }
        )
        fields = response.model_dump()
        model_score = (
            clamp_score(fields.get("research_value_score", 0.0)) * 0.45
            + clamp_score(fields.get("personal_relevance_score", 0.0)) * 0.55
        )
        final_score = clamp_score(heuristic_score * 0.65 + model_score * 0.35)
        reason = fields.get("importance_reason") or "Importance scored by preference heuristic and model judgment."
        signals = fields.get("key_signals") or []
        if signals:
            reason = f"{reason} Signals: {', '.join(signals[:5])}"
    except Exception as e:
        print(f"Importance scoring error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        final_score = heuristic_score
        reason = "Flash importance scoring failed; used configured preference heuristic fallback."

    ai["importance_score"] = round(final_score, 2)
    ai["importance_level"] = importance_level(final_score)
    ai["importance_reason"] = reason
    ai["deep_read_selected"] = False
    ai["deep_read_rank"] = None
    item["AI"] = ai
    return item


def score_importance_for_items(data: List[Dict], model_name: str, max_workers: int, importance_config: Dict) -> List[Dict]:
    if not data:
        return data

    importance_system = (
        "You are a fast research paper importance rater. Score the paper for a daily personal research briefing. "
        "Respect the user's configured priorities more than generic popularity. Return only structured fields."
    )
    importance_template = (
        "User importance configuration:\n{importance_config}\n\n"
        "Paper metadata:\n"
        "Title: {title}\n"
        "Authors: {authors}\n"
        "arXiv categories: {categories}\n"
        "Primary direction: {direction}\n"
        "Subtopic: {subtopic}\n\n"
        "TL;DR:\n{tldr}\n\n"
        "Abstract:\n{abstract}\n\n"
        "Task: rate the paper's importance for the user's daily reading queue."
    )
    llm = build_llm(model_name, ImportanceStructure)
    print("Importance model:", model_name, file=sys.stderr)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(importance_system),
            HumanMessagePromptTemplate.from_template(template=importance_template),
        ]
    )
    chain = prompt_template | llm
    importance_config_text = compact_importance_config_for_prompt(importance_config)

    scored_data = [{} for _ in data]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, item in enumerate(data):
            heuristic_score = heuristic_importance_score(item, importance_config)
            future = executor.submit(score_single_importance, chain, item, importance_config_text, heuristic_score)
            future_to_idx[future] = idx

        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Scoring importance"):
            idx = future_to_idx[future]
            try:
                scored_data[idx] = future.result()
            except Exception as e:
                print(f"Importance worker error at index {idx}: {e}", file=sys.stderr)
                item = data[idx]
                ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
                fallback_score = heuristic_importance_score(item, importance_config)
                ai["importance_score"] = round(fallback_score, 2)
                ai["importance_level"] = importance_level(fallback_score)
                ai["importance_reason"] = "Unhandled importance worker error; used configured preference heuristic fallback."
                ai["deep_read_selected"] = False
                ai["deep_read_rank"] = None
                item["AI"] = ai
                scored_data[idx] = item

    return [item for item in scored_data if item]


def mark_deep_read_selection(items: List[Dict], top_k: int) -> List[Dict]:
    ranked = sorted(
        items,
        key=lambda item: (
            float((item.get("AI") or {}).get("importance_score") or 0.0),
            item.get("id", ""),
        ),
        reverse=True,
    )
    selected_ids = {item.get("id"): rank for rank, item in enumerate(ranked[:top_k], start=1)}
    for item in items:
        ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
        rank = selected_ids.get(item.get("id"))
        ai["deep_read_selected"] = rank is not None
        ai["deep_read_rank"] = rank
        item["AI"] = ai
    return items


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
    detail_model_name = os.environ.get("DETAIL_MODEL_NAME") or os.environ.get("MODEL_NAME", "deepseek-chat")
    filter_model_name = os.environ.get("FILTER_MODEL_NAME") or "deepseek-v4-flash"
    importance_model_name = os.environ.get("IMPORTANCE_MODEL_NAME") or filter_model_name
    filter_max_workers = int(os.environ.get("FILTER_MAX_WORKERS") or str(args.max_workers))
    detail_max_workers = int(os.environ.get("DETAIL_MAX_WORKERS") or str(args.max_workers))
    importance_max_workers = int(os.environ.get("IMPORTANCE_MAX_WORKERS") or str(filter_max_workers))
    language = os.environ.get("LANGUAGE", "Chinese")
    directions = load_directions(args.directions)
    if not directions:
        raise RuntimeError(f"No semantic directions found in {args.directions}")
    importance_config = load_importance_config(args.directions)

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

    print("Open:", args.data, file=sys.stderr)
    filtered_data = filter_all_items(unique_data, filter_model_name, filter_max_workers, directions)
    max_detail_items = int(os.environ.get("MAX_DETAIL_ITEMS") or "0")
    if max_detail_items > 0:
        filtered_data = filtered_data[:max_detail_items]
    print(f"Filtered {len(filtered_data)} relevant papers from {len(unique_data)} crawled papers.", file=sys.stderr)

    processed_data = process_all_items(filtered_data, detail_model_name, language, detail_max_workers, directions, taxonomy)
    taxonomy = update_taxonomy_from_items(taxonomy, processed_data, directions)
    save_taxonomy(taxonomy, args.taxonomy)
    processed_data = score_importance_for_items(processed_data, importance_model_name, importance_max_workers, importance_config)
    daily_top_k = int(os.environ.get("DAILY_DEEP_READ_TOP_K") or importance_config.get("daily_deep_read_top_k", 5) or 5)
    processed_data = mark_deep_read_selection(processed_data, daily_top_k)

    with open(target_file, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Kept {len(processed_data)} semantically relevant papers", file=sys.stderr)


if __name__ == "__main__":
    main()
