import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTIONS_PATH = REPO_ROOT / "config" / "directions.yaml"
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "data" / "taxonomy.json"


DEFAULT_IMPORTANCE_CONFIG: Dict[str, Any] = {
    "daily_deep_read_top_k": 5,
    "direction_weights": {},
    "priority_subtopics": [],
    "boost_keywords": [],
    "negative_keywords": [],
}


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:idx].rstrip()
    return value.rstrip()


def _parse_scalar(value: str) -> str:
    value = _strip_inline_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_simple_directions_yaml(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by config/directions.yaml."""
    directions: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    list_key: Optional[str] = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped == "directions:":
            continue

        if line.startswith("  - "):
            if current:
                directions.append(current)
            current = {}
            list_key = None
            remainder = line[4:].strip()
            if remainder:
                key, value = remainder.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
            continue

        if current is None:
            continue

        if line.startswith("    ") and not line.startswith("      - "):
            key, value = line[4:].split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                current[key] = _parse_scalar(value)
                list_key = None
            else:
                current[key] = []
                list_key = key
            continue

        if line.startswith("      - ") and list_key:
            current[list_key].append(_parse_scalar(line[8:]))

    if current:
        directions.append(current)

    return {"directions": directions}


def load_semantic_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_DIRECTIONS_PATH)
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            parsed = yaml.safe_load(text)
        except Exception:
            parsed = _parse_simple_directions_yaml(text)

    return parsed if isinstance(parsed, dict) else {}


def load_directions(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    parsed = load_semantic_config(config_path)

    directions = parsed.get("directions", []) if isinstance(parsed, dict) else []
    normalized = []
    for item in directions:
        if not isinstance(item, dict) or not item.get("id") or not item.get("name"):
            continue
        direction = deepcopy(item)
        direction["arxiv_categories"] = list(dict.fromkeys(direction.get("arxiv_categories", [])))
        direction["keywords"] = list(dict.fromkeys(direction.get("keywords", [])))

        canonical = []
        for sub in direction.get("canonical_subtopics") or []:
            if not isinstance(sub, dict) or not sub.get("id") or not sub.get("name"):
                continue
            canonical.append(
                {
                    "id": str(sub["id"]),
                    "name": str(sub["name"]),
                    "description": str(sub.get("description") or ""),
                    "keywords": list(dict.fromkeys(sub.get("keywords") or [])),
                }
            )
        direction["canonical_subtopics"] = canonical
        normalized.append(direction)
    return normalized


def load_importance_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    parsed = load_semantic_config(config_path)
    raw = parsed.get("importance", {}) if isinstance(parsed, dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    config = deepcopy(DEFAULT_IMPORTANCE_CONFIG)
    config.update(raw)

    config["daily_deep_read_top_k"] = int(config.get("daily_deep_read_top_k") or 5)
    config["direction_weights"] = {
        str(key): float(value)
        for key, value in dict(config.get("direction_weights") or {}).items()
    }

    priority_subtopics = []
    for item in config.get("priority_subtopics") or []:
        if isinstance(item, str):
            priority_subtopics.append({"name": item, "weight": 1.2, "keywords": []})
        elif isinstance(item, dict) and item.get("name"):
            normalized = deepcopy(item)
            normalized["weight"] = float(normalized.get("weight") or 1.2)
            normalized["keywords"] = list(dict.fromkeys(normalized.get("keywords") or []))
            priority_subtopics.append(normalized)
    config["priority_subtopics"] = priority_subtopics

    config["boost_keywords"] = list(dict.fromkeys(config.get("boost_keywords") or []))
    config["negative_keywords"] = list(dict.fromkeys(config.get("negative_keywords") or []))
    return config


def direction_map(directions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {direction["id"]: direction for direction in directions}


def recall_categories(directions: Iterable[Dict[str, Any]]) -> List[str]:
    categories: List[str] = []
    for direction in directions:
        categories.extend(direction.get("arxiv_categories", []))
    return list(dict.fromkeys(categories))


def slugify(value: str, prefix: str = "topic") -> str:
    ascii_part = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if ascii_part:
        return ascii_part[:64]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _canonical_seed_subtopic(sub: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": sub["id"],
        "name": sub["name"],
        "description": sub.get("description", ""),
        "keywords": list(sub.get("keywords", [])),
        "examples": [],
        "created_at": None,
        "updated_at": None,
        "paper_count": 0,
        "is_canonical": True,
    }


def empty_taxonomy(directions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "version": 2,
        "updated_at": None,
        "directions": {
            direction["id"]: {
                "id": direction["id"],
                "name": direction["name"],
                "subtopics": [
                    _canonical_seed_subtopic(sub)
                    for sub in direction.get("canonical_subtopics", [])
                ],
            }
            for direction in directions
        },
    }


def ensure_canonical_seeded(taxonomy: Dict[str, Any], directions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Make sure each direction has its canonical subtopics present (idempotent)."""
    for direction in directions:
        entry = taxonomy["directions"].setdefault(
            direction["id"],
            {"id": direction["id"], "name": direction["name"], "subtopics": []},
        )
        existing_ids = {sub.get("id") for sub in entry.get("subtopics", [])}
        existing_names = {(sub.get("name") or "").strip().lower() for sub in entry.get("subtopics", [])}
        for canonical in direction.get("canonical_subtopics", []):
            if canonical["id"] in existing_ids:
                for sub in entry["subtopics"]:
                    if sub.get("id") == canonical["id"]:
                        sub["is_canonical"] = True
                        sub["keywords"] = list(dict.fromkeys((sub.get("keywords") or []) + canonical.get("keywords", [])))
                        if not sub.get("description"):
                            sub["description"] = canonical.get("description", "")
                        break
                continue
            if canonical["name"].strip().lower() in existing_names:
                continue
            entry["subtopics"].append(_canonical_seed_subtopic(canonical))
    return taxonomy


def load_taxonomy(path: Optional[str], directions: List[Dict[str, Any]]) -> Dict[str, Any]:
    taxonomy_path = Path(path or DEFAULT_TAXONOMY_PATH)
    if taxonomy_path.exists():
        try:
            taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            taxonomy = empty_taxonomy(directions)
    else:
        taxonomy = empty_taxonomy(directions)

    taxonomy.setdefault("version", 2)
    taxonomy.setdefault("updated_at", None)
    taxonomy.setdefault("directions", {})
    existing_direction_ids = set(taxonomy.get("directions", {}))
    valid_direction_ids = {direction["id"] for direction in directions}
    stale_direction_ids = existing_direction_ids - valid_direction_ids
    taxonomy["directions"] = {
        direction_id: entry
        for direction_id, entry in taxonomy.get("directions", {}).items()
        if direction_id in valid_direction_ids
    }
    stale_prefixes = tuple(f"{direction_id}_" for direction_id in stale_direction_ids)
    if stale_prefixes:
        for entry in taxonomy["directions"].values():
            entry["subtopics"] = [
                subtopic
                for subtopic in entry.get("subtopics", [])
                if not str(subtopic.get("id", "")).startswith(stale_prefixes)
            ]
    for direction in directions:
        entry = taxonomy["directions"].setdefault(
            direction["id"],
            {"id": direction["id"], "name": direction["name"], "subtopics": []},
        )
        entry["name"] = direction["name"]
        entry.setdefault("subtopics", [])
    ensure_canonical_seeded(taxonomy, directions)
    return taxonomy


def save_taxonomy(taxonomy: Dict[str, Any], path: Optional[str] = None) -> None:
    taxonomy_path = Path(path or DEFAULT_TAXONOMY_PATH)
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_path.write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compact_directions_for_prompt(directions: Iterable[Dict[str, Any]]) -> str:
    lines = []
    for direction in directions:
        keywords = ", ".join(direction.get("keywords", [])[:24])
        categories = ", ".join(direction.get("arxiv_categories", []))
        lines.append(
            f"- id: {direction['id']}\n"
            f"  name: {direction['name']}\n"
            f"  description: {direction.get('description', '')}\n"
            f"  arxiv_categories: {categories}\n"
            f"  keywords: {keywords}"
        )
    return "\n".join(lines)


def compact_taxonomy_for_prompt(taxonomy: Dict[str, Any]) -> str:
    lines = []
    for direction_id, entry in taxonomy.get("directions", {}).items():
        lines.append(f"- direction_id: {direction_id} ({entry.get('name', direction_id)})")
        subtopics = entry.get("subtopics", [])
        if not subtopics:
            lines.append("  subtopics: none yet")
            continue
        sorted_subs = sorted(
            subtopics,
            key=lambda sub: (
                0 if sub.get("is_canonical") else 1,
                -int(sub.get("paper_count", 0) or 0),
                sub.get("name", ""),
            ),
        )
        for subtopic in sorted_subs:
            tag = "[CANONICAL] " if subtopic.get("is_canonical") else ""
            count = int(subtopic.get("paper_count", 0) or 0)
            keywords = ", ".join(subtopic.get("keywords", [])[:8])
            keyword_line = f"\n    keywords: {keywords}" if keywords else ""
            lines.append(
                f"  - {tag}id: {subtopic.get('id')} (papers: {count})\n"
                f"    name: {subtopic.get('name')}\n"
                f"    description: {subtopic.get('description', '')}"
                f"{keyword_line}"
            )
    return "\n".join(lines)


def direction_display(direction_id: str, directions: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping = direction_map(directions)
    direction = mapping.get(direction_id, {})
    return {"id": direction_id, "name": direction.get("name", direction_id)}


def normalize_matched_directions(
    direction_ids: Iterable[str],
    directions: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    valid = direction_map(directions)
    normalized = []
    for direction_id in direction_ids:
        if direction_id in valid:
            normalized.append({"id": direction_id, "name": valid[direction_id]["name"]})
    return normalized


def find_existing_subtopic(direction_entry: Dict[str, Any], subtopic_id: str, name: str) -> Optional[Dict[str, Any]]:
    normalized_name = name.strip().lower()
    for subtopic in direction_entry.get("subtopics", []):
        if subtopic.get("id") == subtopic_id:
            return subtopic
        if normalized_name and subtopic.get("name", "").strip().lower() == normalized_name:
            return subtopic
    return None


def upsert_subtopic(
    taxonomy: Dict[str, Any],
    direction_id: str,
    subtopic_id: str,
    name: str,
    description: str,
    paper: Dict[str, Any],
    now: Optional[str] = None,
) -> Dict[str, Any]:
    now = now or datetime.utcnow().strftime("%Y-%m-%d")
    direction_entry = taxonomy["directions"].setdefault(
        direction_id,
        {"id": direction_id, "name": direction_id, "subtopics": []},
    )
    direction_entry.setdefault("subtopics", [])

    if not subtopic_id:
        subtopic_id = slugify(name or "uncategorized")

    existing = find_existing_subtopic(direction_entry, subtopic_id, name)
    if existing is None:
        existing_ids = {item.get("id") for item in direction_entry["subtopics"]}
        if subtopic_id in existing_ids:
            subtopic_id = f"{subtopic_id}_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:6]}"
        existing = {
            "id": subtopic_id,
            "name": name or "未分类",
            "description": description or "模型未能给出清晰定义的论文类别。",
            "examples": [],
            "created_at": now,
            "updated_at": now,
            "paper_count": 0,
        }
        direction_entry["subtopics"].append(existing)

    existing["updated_at"] = now
    existing["paper_count"] = int(existing.get("paper_count", 0)) + 1
    if description and not existing.get("description"):
        existing["description"] = description

    examples = existing.setdefault("examples", [])
    paper_id = paper.get("id", "")
    if paper_id and not any(example.get("id") == paper_id for example in examples):
        examples.append({"id": paper_id, "title": paper.get("title", "")})
        del examples[:-5]

    taxonomy["updated_at"] = now
    return existing
