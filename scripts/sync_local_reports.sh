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
mkdir -p "${REPORT_DIR}/deep_reads"

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

git ls-tree -r --name-only "$DATA_REF" data > "$tmp_file"

count=0
deep_count=0
latest=""
latest_deep=""
while IFS= read -r path; do
  dir="$(dirname "$path")"
  filename="$(basename "$path")"
  if [ "$dir" = "data" ] && [[ "$filename" == *.md ]]; then
    git show "${DATA_REF}:${path}" > "${REPORT_DIR}/${filename}"
    latest="$filename"
    count=$((count + 1))
  elif [ "$dir" = "data/deep_reads" ] && [[ "$filename" == *.md ]]; then
    git show "${DATA_REF}:${path}" > "${REPORT_DIR}/deep_reads/${filename}"
    latest_deep="$filename"
    deep_count=$((deep_count + 1))
  fi
done < "$tmp_file"

if [ "$count" -eq 0 ] && [ "$deep_count" -eq 0 ]; then
  echo "No Markdown reports found on ${DATA_REF}."
  exit 0
fi

echo "Synced ${count} Markdown report(s) into ${REPORT_DIR}."
if [ -n "$latest" ]; then
  echo "Latest report: ${REPORT_DIR}/${latest}"
fi
echo "Synced ${deep_count} deep-read report(s) into ${REPORT_DIR}/deep_reads."
if [ -n "$latest_deep" ]; then
  echo "Latest deep-read report: ${REPORT_DIR}/deep_reads/${latest_deep}"
fi
