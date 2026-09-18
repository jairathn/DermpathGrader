# DermpathGrader

Streamlit app that grades dermatopathology images along two independent
pathways, each backed by its own ChromaDB literature store and a Claude
vision call.

Migrated off Replit 2026-09-18 and rebuilt to **protocol v2.0** the same
day. `docs/STUDY_DESIGN.md` covers the study; this file covers the code.

## Entry points

```
streamlit run app.py --server.port 5000        # interactive
python build_vector_stores.py --check          # are the source PDFs intact
python build_case_registry.py --scan slides/   # build the 350-case registry
python extract_tiles.py --registry data/case_registry.csv
python make_manifest.py                        # freeze the run configuration
python run_tests.py --dry-run                  # validate without API calls
python run_tests.py --replicates 3             # the batch
python verify_logging.py                       # audit before scoring
python join_and_score.py                       # concordance
python tests/test_smoke.py                     # offline, no key needed
```

Python 3.11. The only required secret is `ANTHROPIC_API_KEY`.

## What protocol v2.0 changed

| | v1 | v2.0 |
|---|---|---|
| Cases | ad hoc | 350: 50 per stratum |
| Melanocytic labels | mild / moderate / severe | + **melanoma** |
| Second melanocytic field | `mpath_grade`, two tiers | **MPATH-Dx v2.0 class** (0/I/II/III/IV) |
| Images per case | 1 | **4** (whole slide, 4x, 10x, 40x) |
| Reader input | JPEG | **.svs whole-slide** |
| Model | `claude-opus-4-5-20251101` | **`claude-opus-5`** |
| max_tokens | 1500 | 8000, streamed |
| temperature | 0.1 | none (rejected by this model family) |
| Output parsing | JSON → regex → keyword inference | **structured outputs**, no fallback |
| Log version | 1.0 (one image) | 2.0 (images array) |

`config.py` holds the constants. `config.PROTOCOL_VERSION` is written
into every log, and both the scorer and the verifier refuse to pool logs
from different protocol versions.

## The two pathways

Everything is duplicated per pathway. There is no shared abstraction over
the grading code, and that is deliberate.

| Concern | CSCC (squamous) | Melanocytic |
|---|---|---|
| Retrieval | `rag_system.py` | `nevi_rag_system.py` |
| Grading call | `image_analyzer.py` | `nevi_analyzer.py` |
| Display helpers | `utils.py` | `nevi_utils.py` |
| Vector store | `chroma_db/` | `chroma_db_nevi/` |
| Collection | `scc_grading_literature` | `nevi_grading_literature` |
| Chunks | 72 | 294 |
| Subquery panel | 7 | 12 |
| Strata | well / moderately / poorly | mild / moderate / severe / melanoma |
| Ground truth | `ground_truth_cscc.csv` | `ground_truth_nevus.csv` |
| Logs | `analysis_logs/CSCC/` | `analysis_logs/Nevus/` |

`config.py`, `image_utils.py` and `grading_logger.py` *are* shared. They
carry no grading logic: constants, pixels and file I/O only. The
prohibition is on unifying the prompts, schemas and retrieval panels.

## MPATH-Dx v2.0

`mpath_dx.py`, from Barnhill RL, Elder DE, Piepkorn MW, et al. *JAMA Netw
Open.* 2023;6(1):e2250613. doi:10.1001/jamanetworkopen.2022.50613

Four classes: I low-grade atypia, II high-grade atypia (**including
melanoma in situ**), III melanoma pT1a (<0.8 mm), IV melanoma ≥pT1b, plus
0 nondiagnostic.

**The thing to know before touching this.** v2.0 created its four classes
by removing v1.0's standalone moderate-atypia class. Class I is
"mild-to-moderate" and Class II is "high-end moderate-to-severe". So the
moderate stratum has no single correct class, and
`expected_classes("moderate")` returns `{I, II}` on purpose. The scorer
counts either as concordant and reports those cases separately. Do not
collapse it to one class in code; if the study needs one class per case,
a dermatopathologist assigns it in the `mpath_dx_v2_reference` column.

Second consequence: melanoma in situ is Class II, the same class as
high-grade dysplasia. On the MPATH-Dx arm, in situ melanoma is not
separable from severe dysplasia. The four-way stratum arm and the
melanoma-detection arm are what distinguish them.

## How retrieval works

- The subquery panels are **fixed and hardcoded**, not derived from the
  image. Every case within a pathway retrieves the same passages, so the
  context block is identical across cases. `verify_logging.py` checks
  this per case.
- Top-5 per subquery, ChromaDB default embedder (all-MiniLM-L6-v2).
- Chunking is 1,000 characters with 200 overlap. Confirmed against the
  rebuilt store: 72 CSCC chunks, matching the original exactly.
- The grading criteria tiers are written into the static prompt scaffold.
  Retrieval supplements the prompt; it is not the source of the decision
  rules. Do not describe this system as though the grade is derived from
  retrieved literature.

## Supporting modules

- `config.py` — study and model constants
- `mpath_dx.py` — MPATH-Dx v2.0 classes, mapping and prompt block
- `image_utils.py` — the single image preprocessing path
- `extract_tiles.py` — .svs → the four magnification JPEGs
- `build_case_registry.py` — the 350-case registry and blinded reader manifest
- `build_vector_stores.py` — rebuild Chroma from the PDFs; `--check` first
- `grading_logger.py` — per-case structured logging
- `make_manifest.py` — builds `run_manifest.json`
- `run_tests.py` — batch runner, `--dry-run` supported
- `join_and_score.py` — joins logs to ground truth (1 CSCC arm, 4 melanocytic)
- `verify_logging.py` — audits manifest and logs, ~101 checks per log
- `tests/test_smoke.py` — offline tests, no API key
- `analyze_images.py`, `generate_examples.py` — report builders

## Landmines

1. ~~`generate_examples.py` Section 4 emits hardcoded demo content.~~
   **Fixed.** The fabricated SCC-001/NEVI-001 cases are gone. Section 4
   now reads real rows from `results/` and prints nothing when there are
   none.

2. **The vector stores were lost in the migration and rebuilt.**
   `chroma_db/` and `chroma_db_nevi/` did not come across. They are
   rebuilt from the PDFs in `attached_assets/`.
   *The CSCC rebuild reproduces exactly*: 72 chunks, matching
   `chunk_id_fingerprint_sha256`, and all 7 subquery top-1 distances
   identical to the manifest, including the 0.5462 reference. Seven of
   17 chunks differ by <0.5% of characters (a newer pypdf), which moves
   `corpus_fingerprint_sha256` but not the retrieval results.
   **The nevus store cannot be rebuilt yet**: `Nevi 2025_1749902667337.pdf`
   (20 pp, 96 of the 294 chunks) is still missing. Run
   `python build_vector_stores.py --check` for current status.
   `build_vector_stores.py` refuses to overwrite an existing store
   without `--force`.

3. ~~Nevus `max_tokens` is 1000.~~ It was 1500, and is now 8000 against a
   larger schema, streamed. Do not lower it.

4. ~~The response parser has a keyword-inference fallback.~~ **Fixed.**
   Both analyzers use `output_config.format` with a JSON schema, so the
   API enforces the shape. A malformed response now raises instead of
   returning an invented grade. The v1 fallback silently defaulted to
   "Moderately Differentiated" / "Moderate Dysplasia", which is
   indistinguishable from a real answer once logged. `verify_logging.py`
   fails any v2.0 log whose `parsing.strategy_used` is not
   `structured_output`.

5. **The UI and batch paths must stay byte-identical.** Both now call
   `image_utils.prepare_case_images()`. v1 had two near-duplicate
   functions with different return shapes. Do not reintroduce a second
   preprocessing path.

6. Real retrieval is noisy. All three grade-specific nevus subqueries
   return the same two chunks, one a reference list and one a title page.
   The CSCC "moderately differentiated" subquery returns an author
   affiliation block. Known, not a new bug.

7. **New: field selection is the weak point of the model arm.** Readers
   pan and zoom a whole slide freely; the model gets four fixed frames.
   Whoever picks the 4x/10x/40x coordinates makes part of the diagnostic
   decision first, and if they know the diagnosis the label leaks into
   the input. `tile_selection_method` records `curated` or `auto` per
   case. Do not mix methods within a stratum — that confounds selection
   method with grade. See the header of `extract_tiles.py`.

## Data and assets

- `attached_assets/` — the source literature PDFs (5 of 6 present)
- `data/case_registry.csv` — the 350 cases, both arms
- `data/tile_coords.csv` — curated field coordinates, human-filled
- `data/reader_manifest.csv` — blinded per-reader reading order
- `analysis_logs/` — run output, keyed by pathway
- `run_manifest.json` — the manifest from the last logging run

## Conventions

- Do not add a shared abstraction over the two pathways' grading logic
  without asking. The duplication is load-bearing.
- Any change that could alter retrieval output or the prompt needs a
  `verify_logging.py` run and a note of the before/after distances.
- Bump `config.PROTOCOL_VERSION` whenever the prompt scaffold, output
  schema, magnification set or model changes. Logs carry it, and mixing
  protocol versions in one analysis is silently wrong otherwise.
- Never hand-edit a case log. Re-run; replicates increment automatically.
