#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

REMOTE="${REMOTE:-origin}"
DATA_BRANCH="${DATA_BRANCH:-data}"
DATA_REF="${REMOTE}/${DATA_BRANCH}"
REPORT_DIR="${REPORT_DIR:-local_reports}"

echo "Fetching ${DATA_REF}..."
git fetch "$REMOTE" "$DATA_BRANCH"

mkdir -p "$REPORT_DIR"

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

git ls-tree -r --name-only "$DATA_REF" data > "$tmp_file"

count=0
latest=""
while IFS= read -r path; do
  case "$path" in
    data/*.md)
      filename="$(basename "$path")"
      git show "${DATA_REF}:${path}" > "${REPORT_DIR}/${filename}"
      latest="$filename"
      count=$((count + 1))
      ;;
  esac
done < "$tmp_file"

if [ "$count" -eq 0 ]; then
  echo "No Markdown reports found on ${DATA_REF}."
  exit 0
fi

echo "Synced ${count} Markdown report(s) into ${REPORT_DIR}."
echo "Latest report: ${REPORT_DIR}/${latest}"
