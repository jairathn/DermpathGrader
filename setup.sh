#!/usr/bin/env bash
# One-time local setup for SquamousCancerGrader after the Replit export.
set -euo pipefail

echo "==> Removing Replit environment leftovers if present"
rm -rf .cache .pythonlibs .local .upm __pycache__

if [ ! -f .env ]; then
  echo "==> Creating .env from template (add your key)"
  cp .env.example .env
fi

echo "==> Creating virtualenv"
python3.11 -m venv .venv || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

if command -v uv >/dev/null 2>&1; then
  echo "==> uv found, syncing from uv.lock (exact pins)"
  uv sync
else
  echo "==> uv not found, installing from requirements.txt (approximate)"
  echo "    For exact pins: pip install uv && uv sync"
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo
echo "==> Sanity check: vector stores present?"
for d in chroma_db chroma_db_nevi; do
  if [ -d "$d" ]; then
    echo "    $d OK ($(du -sh "$d" | cut -f1))"
  else
    echo "    $d MISSING - retrieval will not reproduce. See CLAUDE.md."
  fi
done

echo
echo "Done. Run with:  streamlit run app.py"
