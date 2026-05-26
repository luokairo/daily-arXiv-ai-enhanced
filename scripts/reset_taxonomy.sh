#!/usr/bin/env bash
# Reset data branch's data/taxonomy.json so the next workflow run rebuilds
# the taxonomy starting from the canonical seeds defined in
# config/directions.yaml.
#
# Safe to run multiple times. Will prompt before pushing.
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

REMOTE="${REMOTE:-origin}"
DATA_BRANCH="${DATA_BRANCH:-data}"

echo "Fetching ${REMOTE}/${DATA_BRANCH}..."
git fetch "$REMOTE" "$DATA_BRANCH"

if ! git ls-tree -r --name-only "${REMOTE}/${DATA_BRANCH}" | grep -q "^data/taxonomy.json$"; then
  echo "No data/taxonomy.json on ${REMOTE}/${DATA_BRANCH} — nothing to reset."
  exit 0
fi

WORKTREE_DIR="$(mktemp -d -t taxonomy-reset-XXXXXX)"
trap 'git worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true; rm -rf "$WORKTREE_DIR"' EXIT

echo "Creating temporary worktree at $WORKTREE_DIR ..."
git worktree add --detach "$WORKTREE_DIR" "${REMOTE}/${DATA_BRANCH}"

cd "$WORKTREE_DIR"
git checkout -B "$DATA_BRANCH"

if [ ! -f data/taxonomy.json ]; then
  echo "Worktree has no data/taxonomy.json — nothing to reset."
  exit 0
fi

echo
echo "About to delete data/taxonomy.json on branch '${DATA_BRANCH}'."
read -r -p "Proceed and push to ${REMOTE}/${DATA_BRANCH}? [y/N] " confirm
case "$confirm" in
  y|Y|yes|YES) ;;
  *)
    echo "Aborted. No changes pushed."
    exit 1
    ;;
esac

git rm data/taxonomy.json
git -c user.email="$(git config user.email)" \
    -c user.name="$(git config user.name)" \
    commit -m "chore: reset taxonomy for canonical re-seed"
git push "$REMOTE" "$DATA_BRANCH"

echo
echo "Done. Next workflow run will seed taxonomy from config/directions.yaml canonical_subtopics."
