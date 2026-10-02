import argparse
import json
import os
import sys
from itertools import count
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from semantic_arxiv import load_directions

def report_group(item):
    if item.get("report_level", "detail") == "brief":
        return "secondary_brief"
    return "secondary_detail" if item.get("interest_tier") == "secondary" else "primary_detail"

REPORT_GROUPS = [("primary_detail", "主要方向详细摘要"), ("secondary_detail", "辅助方向精选摘要"), ("secondary_brief", "辅助方向简讯")]

def paper_rank(item):
    ai = item.get("AI") or {}
    return (-float(ai.get("importance_rank_score", ai.get("importance_score", 0)) or 0), -float(ai.get("local_priority_score", 0) or 0), str(item.get("id", "")))



def direction_value(item, valid_directions=None):
    valid_directions = valid_directions or {}

    direction = item.get("primary_direction")
    if isinstance(direction, dict):
        direction_id = direction.get("id", "")
        if not valid_directions or direction_id in valid_directions:
            configured = valid_directions.get(direction_id, {})
            return direction_id, configured.get("name") or direction.get("name", direction_id)
    if isinstance(direction, str) and direction:
        if not valid_directions or direction in valid_directions:
            configured = valid_directions.get(direction, {})
            return direction, configured.get("name") or direction

    for matched in item.get("matched_directions", []) or []:
        if not isinstance(matched, dict):
            continue
        direction_id = matched.get("id", "")
        if direction_id in valid_directions:
            return direction_id, valid_directions[direction_id].get("name", direction_id)

    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    candidate_ids = [ai.get("primary_direction_id", "")]
    candidate_ids.extend(ai.get("matched_direction_ids", []) or [])
    for direction_id in candidate_ids:
        if direction_id in valid_directions:
            return direction_id, valid_directions[direction_id].get("name", direction_id)

    if valid_directions:
        return "uncategorized", "未分类"

    categories = item.get("categories", ["Uncategorized"])
    if isinstance(categories, list):
        return categories[0], categories[0]
    return str(categories), str(categories)


def subtopic_value(item):
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    return (
        ai.get("subtopic_id") or "uncategorized",
        ai.get("subtopic_name") or "未分类",
        ai.get("subtopic_description") or "",
    )


def safe_authors(authors):
    if isinstance(authors, list):
        return ", ".join(authors)
    return authors or ""


def render_paper(template, item, idx, valid_directions=None):
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    direction_id, direction_name = direction_value(item, valid_directions)
    subtopic_id, subtopic_name, _ = subtopic_value(item)
    deep_read_rank = ai.get("deep_read_rank")
    deep_read_badge = f"Top {deep_read_rank} 精读候选" if ai.get("deep_read_selected") and deep_read_rank else "未入选精读"
    if item.get('report_level', 'detail') == 'brief':
        return (f"### [{idx}] [{item.get('title', 'Untitled')}]({item.get('abs') or item.get('pdf')})\n"
                f"*{safe_authors(item.get('authors'))}*\n\n"
                f"方向：{direction_name} · 相关性：{item.get('relevance_status', 'uncertain')}\n\n"
                f"> 简讯，未做详细分析\n\n{ai.get('tldr', '')}\n\n"
                f"筛选理由：{ai.get('classification_reason', '')}\n\n"
                f"初步阅读优先级：{ai.get('importance_score', '')}")
    return template.format(
        title=item.get("title", "Untitled"),
        authors=safe_authors(item.get("authors")),
        summary=item.get("summary", ""),
        url=item.get("abs") or item.get("pdf") or f"https://arxiv.org/abs/{item.get('id', '')}",
        tldr=ai.get("tldr", item.get("summary", "")),
        motivation=ai.get("motivation", ""),
        method=ai.get("method", ""),
        result=ai.get("result", ""),
        conclusion=ai.get("conclusion", ""),
        classification_reason=ai.get("classification_reason", ""),
        importance_score=ai.get("importance_score", ""),
        importance_level=ai.get("importance_level", ""),
        importance_reason=ai.get("importance_reason", ""),
        deep_read_badge=deep_read_badge,
        direction=direction_name,
        direction_id=direction_id,
        subtopic=subtopic_name,
        subtopic_id=subtopic_id,
        cate=", ".join(item.get("categories", [])) if isinstance(item.get("categories"), list) else item.get("categories", ""),
        idx=idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the jsonline file")
    parser.add_argument(
        "--directions",
        type=str,
        default=str(ROOT_DIR / "config" / "directions.yaml"),
        help="Path to semantic direction config",
    )
    args = parser.parse_args()

    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    directions = load_directions(args.directions)
    direction_order = {direction["id"]: idx for idx, direction in enumerate(directions)}
    valid_directions = {direction["id"]: direction for direction in directions}

    template = Path(__file__).with_name("paper_template.md").read_text(encoding="utf-8")
    groups = {key: [] for key, _ in REPORT_GROUPS}
    for item in data:
        groups[report_group(item)].append(item)
    markdown = "<div id=toc></div>\n\n# Table of Contents\n\n"
    for key, label in REPORT_GROUPS:
        if groups[key]:
            markdown += f"- [{label}](#report-{key}) [Total: {len(groups[key])}]\n"
    idx = count(1)
    for key, label in REPORT_GROUPS:
        if not groups[key]:
            continue
        markdown += f"\n\n<div id='report-{key}'></div>\n\n# {label} [[Back]](#toc)\n\n"
        markdown += "\n\n".join(render_paper(template, item, next(idx), valid_directions)
                                   for item in sorted(groups[key], key=paper_rank))

    data_path = Path(args.data)
    output_path = str(data_path.with_name(data_path.name.split("_")[0] + ".md"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Wrote {output_path}")
