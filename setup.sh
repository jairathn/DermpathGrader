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
echo "==> Sanity check: source PDFs and vector stores"
python build_vector_stores.py --check || true

for d in chroma_db chroma_db_nevi; do
  if [ -d "$d" ]; then
    echo "    $d present ($(du -sh "$d" | cut -f1))"
  else
    echo "    $d absent - build it with: python build_vector_stores.py"
  fi
done

echo
echo "==> Offline tests (no API key needed)"
python tests/test_smoke.py || true

echo
echo "Done. Run with:  streamlit run app.py"
echo "Study design: docs/STUDY_DESIGN.md   Code map: CLAUDE.md"
