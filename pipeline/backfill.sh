#!/usr/bin/env bash
#
# backfill.sh — the one-time history run.
#
# For every creator in creators.txt: scrape their last 40 posts, keep the ones
# that are actually about fitness or nutrition, and analyse ALL of them — not
# the top 3. The 3 was a cost guard for the POC and nothing else.
#
#   ./pipeline/backfill.sh                  every creator in creators.txt
#   ./pipeline/backfill.sh fit.khurana      just one
#   ./pipeline/backfill.sh --limit 60       look further back than 40 posts
#
# Every stage caches per creator, so stopping this and re-running it costs
# nothing for work already done. It is safe to run again after a failure.
#
# When it finishes it rebuilds the website's data automatically, so the last
# thing you see is the archive the site will serve.

set -uo pipefail
cd "$(dirname "$0")/.."

BOLD=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'; RST=$'\033[0m'

CREATORS_FILE="pipeline/creators.txt"

# Anything that isn't a flag is treated as an explicit creator handle.
handles=()
flags=()
for a in "$@"; do
  case "$a" in
    -*) flags+=("$a") ;;
    *)  handles+=("$a") ;;
  esac
done

if [ ${#handles[@]} -eq 0 ]; then
  if [ ! -f "$CREATORS_FILE" ]; then
    echo "${RED}No $CREATORS_FILE and no handles given.${RST}"
    echo "Add one handle per line to $CREATORS_FILE, or pass them on the command line."
    exit 1
  fi
  # Strip comments and blank lines.
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | tr -d '[:space:]')"
    [ -n "$line" ] && handles+=("$line")
  done < "$CREATORS_FILE"
fi

if [ ${#handles[@]} -eq 0 ]; then
  echo "${RED}No creators to run.${RST}"; exit 1
fi

echo
echo "${BOLD}Backfill${RST}  ${DIM}${#handles[@]} creator(s), last 40 posts each, every on-topic reel analysed${RST}"
echo "${DIM}  $(printf '@%s ' "${handles[@]}")${RST}"
echo

# --- environment ------------------------------------------------------------
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -z "$PY" ] && { echo "${RED}Python 3 is not installed.${RST}  brew install python"; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || {
  echo "${RED}ffmpeg is not installed${RST} — needed for frames and audio."
  echo "  brew install ffmpeg"; exit 1; }

VENV=".venv"
if [ ! -d "$VENV" ]; then
  echo "${DIM}Setting up a local Python environment (about 20 seconds)…${RST}"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet requests
  echo "${GRN}✓${RST} environment ready"
fi
"$VENV/bin/python" -c "import requests" >/dev/null 2>&1 || "$VENV/bin/pip" install --quiet requests

[ -f ".env" ] || { echo "${YEL}No .env file.${RST}  cp .env.example .env  then paste your keys in."; exit 1; }

# --- run --------------------------------------------------------------------
started=$(date +%s)
failed=()
for h in "${handles[@]}"; do
  echo
  echo "${BOLD}────────────────────────────────────────────────────────${RST}"
  echo "${BOLD}  @$h${RST}"
  echo "${BOLD}────────────────────────────────────────────────────────${RST}"
  # --top all is the default now, but it is stated here so the intent of this
  # script is obvious from the command it runs.
  if ! "$VENV/bin/python" pipeline/videolens.py "$h" --top all --limit 40 "${flags[@]+"${flags[@]}"}"; then
    echo "${RED}✗ @$h did not finish${RST}"
    failed+=("$h")
  fi
done

elapsed=$(( $(date +%s) - started ))
echo
echo "${BOLD}────────────────────────────────────────────────────────${RST}"
printf "${BOLD}Backfill finished${RST} in %dm %ds\n" $((elapsed/60)) $((elapsed%60))
if [ ${#failed[@]} -gt 0 ]; then
  echo "${YEL}Did not finish:${RST} $(printf '@%s ' "${failed[@]}")"
  echo "${DIM}Re-run just those handles; everything already done is cached.${RST}"
fi

# --- publish ----------------------------------------------------------------
echo
echo "${BOLD}Rebuilding the website's data${RST}"
"$VENV/bin/python" pipeline/export_hub.py || exit 1

echo "${DIM}Commit the result and push:${RST}"
echo "  git add -A && git commit -m 'Backfill archive' && git push"
