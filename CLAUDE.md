# SquamousCancerGrader

Streamlit app that grades dermatopathology images along two independent
pathways, each backed by its own ChromaDB literature store and a Claude
vision call.

Migrated off Replit 2026-09-18. Anything in this file marked **verify**
came from prior working notes rather than a fresh read of the code, so
confirm it before you rely on it.

## Entry point

```
streamlit run app.py --server.port 5000
```

Python 3.11. The only required secret is `ANTHROPIC_API_KEY`; several
modules raise on startup if it is unset.

## The two pathways

Everything is duplicated per pathway. There is no shared abstraction
layer, so a change to one side does not propagate to the other. This is
the single most important thing to know about the codebase.

| Concern | CSCC (squamous) | Nevus (melanocytic) |
|---|---|---|
| Retrieval | `rag_system.py` | `nevi_rag_system.py` |
| Grading call | `image_analyzer.py` | `nevi_analyzer.py` |
| Helpers | `image_utils.py` | `nevi_utils.py` |
| Vector store dir | `chroma_db/` | `chroma_db_nevi/` |
| Collection | `scc_grading_literature` | `nevi_grading_literature` |
| Chunks | 72 | 294 |
| Subquery panel | 7 queries | 12 queries |
| Ground truth | `ground_truth_cscc.csv` | `ground_truth_nevus.csv` |
| Logs | `analysis_logs/CSCC/` | `analysis_logs/Nevus/` |

CSCC grades well / moderately / poorly differentiated. Nevus grades
mild / moderate / severe dysplasia plus an MPATH-Dx low- vs high-grade
mapping.

## How retrieval actually works

Worth being precise about, because it is easy to describe wrongly:

- The subquery panels are **fixed and hardcoded**, not derived from the
  image. Every case within a pathway retrieves the same passages.
  The retrieved context block is therefore identical across cases.
- Top-5 per subquery, ChromaDB default embedder (all-MiniLM-L6-v2).
- Chunking is 1,000 characters with 200 overlap. **verify**
- The grading criteria tiers are also written directly into the static
  prompt scaffold. Retrieval supplements the prompt; it is not the
  source of the decision rules. Do not describe this system as though
  the grade is derived from retrieved literature.

## Supporting modules

- `utils.py` — shared helpers
- `grading_logger.py` — per-case structured logging
- `make_manifest.py` — builds `run_manifest.json`
- `run_tests.py` — batch runner across a case set
- `join_and_score.py` — joins logs to ground truth, computes concordance
  (three separate scoring arms on the nevus side)
- `verify_logging.py` — audits the manifest and case logs against the
  logging spec, ~110 checks. Run it before and after any batch.
- `analyze_images.py`, `generate_examples.py` — one-off report builders

## Landmines

Read these before touching anything that feeds a manuscript or supplement.

1. **`generate_examples.py` Section 4 emits hardcoded demo content.**
   Cases SCC-001/002 and NEVI-001/002 are fabricated illustrations, not
   real retrieval output. The snippets contradict the real Section 2
   output, carry no distance/source/page, and miscite MPATH-Dx. This
   output must never reach a supplement or a reviewer. Fix it or delete
   the section.
2. **The vector stores are reproducibility artifacts.** Published
   retrieval distances only reproduce against these exact directories.
   Reference values: CSCC subquery-1 top-1 = 0.5462, nevus = 0.4066.
   Rebuilding the stores from the PDFs in `attached_assets/` will shift
   every distance. Never regenerate them casually, and never commit a
   regenerated store over the original.
3. **Nevus `max_tokens` is 1000 against an 8-field JSON schema.** This
   can truncate mid-response. **verify** current value.
4. **The response parser has a keyword-inference fallback** that can
   return a grade even when no valid JSON came back. A silent fallback
   looks identical to a successful parse in the logs. Check whether the
   log records which path produced the grade.
5. **`process_image_file()` must byte-match the Streamlit upload path.**
   If the batch runner preprocesses differently than the UI, batch
   results are not comparable to interactive results.
6. Real retrieval is noisy. All three grade-specific nevus subqueries
   return the same two chunks, one of which is a reference list and one
   a title page. The CSCC "moderately differentiated" subquery returns
   an author affiliation block. This is known, not a new bug.

## Data and assets

- `attached_assets/` — the source literature PDFs the stores were built
  from, plus test images
- `analysis_logs/` — prior run output, keyed by pathway
- `run_manifest.json` — the built manifest from the last logging run

## Not carried over from Replit

`.replit` and the workflow definitions are kept for reference only and
do nothing outside Replit. Same for the nix package list, which existed
to satisfy Pillow's image codecs (freetype, lcms2, libimagequant,
libjpeg, libtiff, libwebp, openjpeg, zlib). On macOS the Pillow wheels
bundle these. On Linux install the distro equivalents if Pillow fails
to import.

Replit held `ANTHROPIC_API_KEY` as a platform Secret, so it is not in
this tree. Put it in `.env`.

## Conventions

- Do not add a shared abstraction over the two pathways without asking.
  The duplication is load-bearing for reproducibility: the CSCC arm is
  what the published results ran on.
- Any change that could alter retrieval output or the prompt needs a
  `verify_logging.py` run and a note of the before/after distances.
