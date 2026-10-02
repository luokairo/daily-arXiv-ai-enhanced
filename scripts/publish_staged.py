"""Promote a validated run's staged artifacts (no network or Git operations)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))
from runtime import atomic_write


def publish(source, destination):
    source, destination = Path(source), Path(destination)
    # Read and validate the entire batch before replacing any existing artifact.
    pending = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            json.loads(content)
        elif path.suffix == ".jsonl":
            for line in content.splitlines():
                if line.strip():
                    json.loads(line)
        if path.name.endswith("-usage.json"):
            date = next(source.glob("*_AI_enhanced_*.jsonl")).name.split("_")[0]
            relative = Path("run_metrics") / f"{date}-{path.name}"
        pending.append((destination / relative, content))
    if not pending or not (source / "taxonomy.json").exists() or not list(source.glob("*_AI_enhanced_*.jsonl")):
        raise RuntimeError("Incomplete staging directory; refusing publication")
    for target, content in pending:
        atomic_write(target, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    publish(args.source, args.destination)
