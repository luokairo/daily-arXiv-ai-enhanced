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


def direction_value(item):
    direction = item.get("primary_direction")
    if isinstance(direction, dict):
        return direction.get("id", ""), direction.get("name", direction.get("id", ""))
    if isinstance(direction, str) and direction:
        return direction, direction
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


def render_paper(template, item, idx):
    ai = item.get("AI", {}) if isinstance(item.get("AI"), dict) else {}
    direction_id, direction_name = direction_value(item)
    subtopic_id, subtopic_name, _ = subtopic_value(item)
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

    grouped = {}
    for item in data:
        direction_id, direction_name = direction_value(item)
        subtopic_id, subtopic_name, subtopic_description = subtopic_value(item)
        direction_group = grouped.setdefault(
            direction_id,
            {
                "name": direction_name,
                "order": direction_order.get(direction_id, len(direction_order)),
                "subtopics": {},
            },
        )
        subtopic_group = direction_group["subtopics"].setdefault(
            subtopic_id,
            {
                "name": subtopic_name,
                "description": subtopic_description,
                "papers": [],
            },
        )
        subtopic_group["papers"].append(item)

    template = open("paper_template.md", "r", encoding="utf-8").read()

    markdown = "<div id=toc></div>\n\n# Table of Contents\n\n"
    sorted_directions = sorted(grouped.items(), key=lambda item: (item[1]["order"], item[1]["name"]))

    for direction_id, direction in sorted_directions:
        paper_total = sum(len(subtopic["papers"]) for subtopic in direction["subtopics"].values())
        anchor = f"direction-{direction_id}"
        markdown += f"- [{direction['name']}](#{anchor}) [Total: {paper_total}]\n"
        for subtopic_id, subtopic in sorted(direction["subtopics"].items(), key=lambda item: item[1]["name"]):
            markdown += f"  - [{subtopic['name']}](#{anchor}-{subtopic_id}) [Total: {len(subtopic['papers'])}]\n"

    idx = count(1)
    for direction_id, direction in sorted_directions:
        anchor = f"direction-{direction_id}"
        markdown += f"\n\n<div id='{anchor}'></div>\n\n# {direction['name']} [[Back]](#toc)\n\n"
        for subtopic_id, subtopic in sorted(direction["subtopics"].items(), key=lambda item: item[1]["name"]):
            markdown += f"\n\n<div id='{anchor}-{subtopic_id}'></div>\n\n## {subtopic['name']} [[Back]](#toc)\n\n"
            if subtopic.get("description"):
                markdown += f"{subtopic['description']}\n\n"
            papers = [render_paper(template, paper, next(idx)) for paper in subtopic["papers"]]
            markdown += "\n\n".join(papers)

    data_path = Path(args.data)
    output_path = str(data_path.with_name(data_path.name.split("_")[0] + ".md"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Wrote {output_path}")
