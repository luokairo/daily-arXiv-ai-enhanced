"""Run a bounded historical sample through interest allocation; never publish it."""
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
    parser.add_argument('--sample-limit', type=int, default=1)
    parser.add_argument('--secondary-detail-limit', type=int, default=10)
    parser.add_argument('--deep-read-top-k', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.sample_limit <= 20 or not 0 <= args.secondary_detail_limit <= 20 or not 1 <= args.deep_read_top_k <= 3:
        parser.error('Diagnostic bounds: 1-20 papers, 0-20 secondary details, 1-3 deep reads')
    papers = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    directions = enhance.load_directions()
    importance = enhance.load_importance_config()
    wanted = {key.strip() for key in args.paper_id.split(',') if key.strip()}
    papers = list({p['id']: p for p in papers if not wanted or p.get('id') in wanted}.values())
    if wanted - {p['id'] for p in papers}:
        raise SystemExit('Some requested paper IDs are absent from this historical file')
    if not papers:
        raise SystemExit("No matching candidate in this historical file")
    papers = sorted(papers, key=lambda p: (-enhance.candidate_priority(p, directions, importance), p["id"]))[:args.sample_limit]
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output.resolve() / "sample.jsonl"
    source.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in papers), encoding="utf-8")
    env = dict(os.environ, AI_OUTPUT_DIR=str(args.output.resolve()),
               SELECTION_MODE="semantic",
               AI_CACHE_DIR=str(args.output.resolve() / "cache"),
               MAX_AI_CANDIDATES=str(len(papers)), MAX_DETAIL_ITEMS=str(len(papers)), SECONDARY_DETAIL_LIMIT=str(args.secondary_detail_limit),
               DAILY_DEEP_READ_TOP_K=str(args.deep_read_top_k), ENABLE_DEEP_READ="true",
               USE_MODEL_FILTER="false", USE_MODEL_IMPORTANCE="false", DETAIL_MAX_WORKERS="1",
               DEEP_READ_MAX_WORKERS="1", LANGUAGE="Chinese")
    env["AI_METRICS_PATH"] = str(args.output.resolve() / "enhance-usage.json")
    subprocess.run([sys.executable, str(ROOT / "ai/enhance.py"), "--data", str(source),
                    "--taxonomy", str(args.taxonomy.resolve())], env=env, check=True)
    enhanced = args.output.resolve() / "sample_AI_enhanced_Chinese.jsonl"
    subprocess.run([sys.executable, str(ROOT / "to_md/convert.py"), '--data', str(enhanced)], env=env, check=True)
    env["AI_METRICS_PATH"] = str(args.output.resolve() / "deep-read-usage.json")
    subprocess.run([sys.executable, str(ROOT / "ai/deep_read.py"), "--data", str(enhanced),
                    "--output", str(args.output.resolve() / "deep-read.md"),
                    "--output-json", str(args.output.resolve() / "deep-read.json")], env=env, check=True)
    print("Diagnostic complete. No data branch or Pages publication was performed.")


if __name__ == "__main__":
    main()
