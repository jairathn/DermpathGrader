# DermpathGrader

Grades dermatopathology images along two independent pathways, each
backed by its own literature retrieval store and a Claude vision call:

- **CSCC** — well / moderately / poorly differentiated
- **Melanocytic** — MPATH-Dx v2.0 Class I / II / III / IV, with melanoma
  distinguished as in situ or invasive and subtyped

Protocol v2.0: 350 cases (50 per stratum — 4 melanocytic classes, 3 CSCC
grades), four magnifications per case (whole slide, 4x, 10x, 40x),
`claude-opus-5`.

Dermatopathologist readers grade whole-slide `.svs`; the model grades
JPEG derivatives, because the vision API cannot read `.svs`.

## Quick start

```bash
./setup.sh                              # venv, deps, source-PDF check
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python doctor.py --ping                 # everything ready? model reachable?
python tests/test_smoke.py              # offline, no key needed
python tests/test_e2e.py                # whole chain, offline
streamlit run app.py
```

## Batch run

```bash
python build_vector_stores.py --check   # are the source PDFs intact
python build_vector_stores.py           # build the stores
python build_case_registry.py --scan slides/
python extract_tiles.py --registry data/case_registry.csv
python make_manifest.py --yes
python run_tests.py --dry-run           # validate + cost estimate, no API spend
python run_tests.py --workers 4         # resumable: re-run to fill gaps
python verify_logging.py
python join_and_score.py                # concordance, replicates, reader study
python report.py --all                  # synoptic report per case
```

## Documentation

- `CLAUDE.md` — code map, module layout, and the landmines to read before
  changing anything that feeds a manuscript
- `docs/STUDY_DESIGN.md` — the study, the two-arm asymmetry, the MPATH-Dx
  v2.0 tension with the moderate stratum, and the scoring arms
- `docs/DEPLOYMENT.md` — hosting options, and why GitHub Pages cannot run
  this
- `MIGRATION.md` — how the code came off Replit and what was lost

## Reliability

The API is called in exactly one place (`claude_transport.py`): bounded
retries with backoff on transient failures, refusal and truncation
raised as errors and logged with their cause, no model fallbacks, every
attempt recorded in the case log. The prompt scaffold is cached
server-side, so a 1,000-call batch pays for it once per cache window.
`run_tests.py` is resumable and refuses to run against a stale manifest.
`tests/test_e2e.py` drives the real runner, verifier, scorer and report
renderer against a scripted API double, including a transient failure,
a refusal and a disagreeing replicate, with no key and no stores.

## Status

- CSCC vector store rebuilt; all 7 published retrieval distances
  reproduce exactly. See `chroma_db/PROVENANCE.md`.
- Melanocytic store blocked on one missing source PDF
  (`Nevi 2025_1749902667337.pdf`).

Research and educational use only. Not for clinical diagnosis.
