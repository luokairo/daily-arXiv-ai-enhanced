"""Run at most one paper through summary and deep reading; never publish it."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai"))
import enhance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--paper-id", default="")
    args = parser.parse_args()
    papers = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    directions = enhance.load_directions()
    importance = enhance.load_importance_config()
    keywords = enhance.collect_filter_keywords(directions, importance)
    papers = [paper for paper in papers if (paper.get("id") == args.paper_id if args.paper_id else
              enhance.local_filter_item(paper, directions, keywords)["state"] != "drop")]
    if not papers:
        raise SystemExit("No matching candidate in this historical file")
    paper = sorted(papers, key=lambda p: (-enhance.candidate_priority(p, directions, importance), p["id"]))[0]
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output.resolve() / "sample.jsonl"
    source.write_text(json.dumps(paper, ensure_ascii=False) + "\n", encoding="utf-8")
    env = dict(os.environ, AI_OUTPUT_DIR=str(args.output.resolve()),
               AI_CACHE_DIR=str(args.output.resolve() / "cache"),
               MAX_DETAIL_ITEMS="1", DAILY_DEEP_READ_TOP_K="1", ENABLE_DEEP_READ="true",
               USE_MODEL_FILTER="false", USE_MODEL_IMPORTANCE="false", DETAIL_MAX_WORKERS="1",
               DEEP_READ_MAX_WORKERS="1", LANGUAGE="Chinese")
    env["AI_METRICS_PATH"] = str(args.output.resolve() / "enhance-usage.json")
    subprocess.run([sys.executable, str(ROOT / "ai/enhance.py"), "--data", str(source),
                    "--taxonomy", str(args.taxonomy.resolve())], env=env, check=True)
    enhanced = args.output.resolve() / "sample_AI_enhanced_Chinese.jsonl"
    env["AI_METRICS_PATH"] = str(args.output.resolve() / "deep-read-usage.json")
    subprocess.run([sys.executable, str(ROOT / "ai/deep_read.py"), "--data", str(enhanced),
                    "--output", str(args.output.resolve() / "deep-read.md"),
                    "--output-json", str(args.output.resolve() / "deep-read.json")], env=env, check=True)
    print("Diagnostic complete. No data branch or Pages publication was performed.")


if __name__ == "__main__":
    main()
