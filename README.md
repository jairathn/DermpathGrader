# DermpathGrader

Grades dermatopathology images along two independent pathways, each
backed by its own literature retrieval store and a Claude vision call:

- **CSCC** — well / moderately / poorly differentiated
- **Melanocytic** — mild / moderate / severe dysplasia, and **melanoma**,
  plus an MPATH-Dx v2.0 class

Protocol v2.0: 350 cases (50 per stratum), four magnifications per case
(whole slide, 4x, 10x, 40x), `claude-opus-5`.

Dermatopathologist readers grade whole-slide `.svs`; the model grades
JPEG derivatives, because the vision API cannot read `.svs`.

## Quick start

```bash
./setup.sh                              # venv, deps, source-PDF check
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python tests/test_smoke.py              # offline, no key needed
streamlit run app.py
```

## Batch run

```bash
python build_vector_stores.py --check   # are the source PDFs intact
python build_vector_stores.py           # build the stores
python build_case_registry.py --scan slides/
python extract_tiles.py --registry data/case_registry.csv
python make_manifest.py
python run_tests.py --dry-run           # validate, no API spend
python run_tests.py --replicates 3
python verify_logging.py
python join_and_score.py
```

## Documentation

- `CLAUDE.md` — code map, module layout, and the landmines to read before
  changing anything that feeds a manuscript
- `docs/STUDY_DESIGN.md` — the study, the two-arm asymmetry, the MPATH-Dx
  v2.0 tension with the moderate stratum, and the scoring arms
- `MIGRATION.md` — how the code came off Replit and what was lost

## Status

- CSCC vector store rebuilt; all 7 published retrieval distances
  reproduce exactly. See `chroma_db/PROVENANCE.md`.
- Melanocytic store blocked on one missing source PDF
  (`Nevi 2025_1749902667337.pdf`).

Research and educational use only. Not for clinical diagnosis.
