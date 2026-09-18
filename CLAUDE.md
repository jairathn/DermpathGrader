# DermpathGrader

Streamlit app and batch pipeline that grade dermatopathology cases along
two independent pathways, each backed by its own ChromaDB literature
store and a Claude vision call.

Migrated off Replit 2026-09-18, rebuilt to protocol v2.0 the same day,
hardened to **protocol v2.1** the same evening. `docs/STUDY_DESIGN.md`
covers the study; this file covers the code.

## Entry points

```
python doctor.py [--ping]                      # is this machine ready?
streamlit run app.py --server.port 5000        # interactive
python build_vector_stores.py --check          # are the source PDFs intact
python build_case_registry.py --scan slides/   # the 350-case registry
python extract_tiles.py --registry data/case_registry.csv
python make_manifest.py --yes [--accept-store-divergence Nevus]
python run_tests.py --dry-run                  # validate + cost, no API calls
python run_tests.py --workers 4                # the batch; resumable
python verify_logging.py                       # audit before scoring
python join_and_score.py                       # concordance -> results/
python report.py --all                         # synoptic reports -> reports/
python tests/test_smoke.py                     # offline unit tests
python tests/test_e2e.py                       # offline end-to-end
```

Python 3.11. The only required secret is `ANTHROPIC_API_KEY`.
`credentials.ensure_api_key()` resolves it from, in order: the
environment, Streamlit secrets, then a gitignored `.env` in the project
root. It is called by `app.py`, `doctor.py` and `run_tests.py` before
any client is built. The key is never committed, in any form.

## Protocol history

| | v1 | v2.0 | v2.1 |
|---|---|---|---|
| Cases | ad hoc | 350: 50 per stratum | same |
| Melanocytic labels | mild / moderate / severe | **MPATH-Dx v2.0 Class I–IV** | same |
| Melanoma | not representable | in situ vs invasive, subtyped, Breslow | + ulceration, mitoses |
| Clinical fields | — | — | **adequacy, differential, ancillary studies; CSCC subtype, depth, Broders, high-risk features** |
| Images per case | 1 | **4** (whole slide, 4x, 10x, 40x) | same |
| Model | `claude-opus-4-5-20251101` | **`claude-opus-5`** | same |
| Prompt placement | user turn | user turn | **cached `system` block** |
| Output parsing | JSON → regex → keyword inference | structured outputs | same |
| Transport | one call, no retry | one call | **retries, refusal/truncation as errors, attempts logged** |
| Log version | 1.0 | 2.0 | **2.1** (+ `failure`, `stop_details`, cache usage) |

`config.PROTOCOL_VERSION` is written into every log. The scorer and the
verifier refuse to pool logs from different protocol versions; the
batch runner refuses to run against a manifest from a different one.

## The two pathways

Everything about *grading* is duplicated per pathway, deliberately.

| Concern | CSCC | Melanocytic |
|---|---|---|
| Retrieval | `rag_system.py` | `nevi_rag_system.py` |
| Prompt, schema, call | `image_analyzer.py` | `nevi_analyzer.py` |
| Display | `utils.py` | `nevi_utils.py` |
| Vector store | `chroma_db/` (72 chunks) | `chroma_db_nevi/` (251 chunks) |
| Strata | well / moderately / poorly | Class I / II / III / IV |
| Ground truth | `ground_truth_cscc.csv` | `ground_truth_nevus.csv` |
| Reader grades | `data/reader_grades_cscc.csv` | `data/reader_grades_nevus.csv` |

Shared modules carry **no grading logic**: `config.py` (constants),
`credentials.py` (key discovery),
`image_utils.py` (pixels), `grading_logger.py` (file I/O),
`claude_transport.py` (the API call itself), `mpath_dx.py` (the
published class definitions), `report.py` (rendering). The prohibition
is on unifying prompts, schemas and retrieval panels.

## How a case is graded

1. `image_utils.prepare_case_images()` — the single preprocessing path,
   used by both the UI and the batch runner. Rejects `.svs`, non-image
   bytes, truncated files, decompression bombs; caps the long edge at
   1568 px; re-encodes to JPEG deterministically; records the quality
   actually used. A missing magnification is a hard error.
2. The analyzer builds the request: the prompt scaffold (criteria,
   MPATH-Dx definitions, retrieved literature) goes in `system` with a
   cache breakpoint because it is identical for every case in the
   pathway; the user turn is four captioned images and one line.
3. `claude_transport.grade()` sends it. Transient failures retry with
   backoff and every failed attempt is recorded. `stop_reason ==
   "refusal"` and `"max_tokens"` raise — a refusal is not a grade and a
   truncated JSON is not parsed. **No model fallbacks**: a batch where
   some cases were graded by a different model is not one experiment.
4. The response is schema-enforced JSON. The analyzer derives
   cross-field consistency flags (e.g. an in situ melanoma placed in
   Class III) and records them; it never corrects the answer.
5. `grading_logger.CaseLogger` writes one log per case-replicate,
   including for failures. Logs are never overwritten.

## MPATH-Dx v2.0

`mpath_dx.py`, from Barnhill RL, Elder DE, Piepkorn MW, et al. *JAMA Netw
Open.* 2023;6(1):e2250613. doi:10.1001/jamanetworkopen.2022.50613

Four classes: I low-grade atypia, II high-grade atypia (**including
melanoma in situ**), III melanoma pT1a (<0.8 mm), IV melanoma ≥pT1b, plus
0 nondiagnostic. **The study samples these classes directly**, 50 each.

`expected_class()` returns one class and raises on a legacy three-tier
label, because v2.0 built its classes by deleting the standalone
moderate category: "moderate" spans I and II and is never auto-converted.
`class_from_dysplasia_grade()` keeps the honest set-valued mapping for
importing old labels; `class_for_melanoma()` places a melanoma from
subtype and Breslow and refuses invasive melanoma with no thickness.

**Class II holds two different things** — high-grade dysplastic nevi and
melanoma in situ. `lesion_category` and `melanoma_subtype` distinguish
them, which is why melanoma detection is scored on `lesion_category`, not
class. `build_case_registry.py` prints the Class II composition.

## Retrieval

- Subquery panels are **fixed and hardcoded**, so the context block is
  identical across cases in a pathway. `run_tests.py` computes it once
  per batch, checks its hash against the manifest before the first API
  call, and hands every worker a frozen read-only copy.
- Top-5 per subquery, all-MiniLM-L6-v2, 1,000-char chunks with 200
  overlap.
- The grading criteria live in the static scaffold. Retrieval
  supplements the prompt; it is not the source of the decision rules.

## Scoring (`join_and_score.py`)

CSCC: exact three-way, with unweighted and quadratic-weighted kappa.
Melanocytic: exact class; Class I vs II+ (re-excision); melanoma
detection (sensitivity/specificity on `lesion_category`); in situ vs
invasive among reference melanomas; Breslow agreement at 0.8 mm.
Both: replicate unanimity and Fleiss' kappa across replicates.
Reader study: model and each reader against the reference, readers
against each other, model against reader consensus — on the same case
set. All proportions carry Wilson intervals. Failed logs (refusal,
truncation) are counted in `parser_summary.csv`, never scored.

## Landmines

1. ~~`generate_examples.py` Section 4 emits fabricated content.~~ Fixed;
   it reads real rows from `results/` and prints nothing otherwise.
2. **The vector stores were lost in the migration and rebuilt.**
   *CSCC reproduces exactly*: 72 chunks, matching fingerprint, all 7
   subquery top-1 distances identical to the manifest. *Nevus does not*:
   `Nevi 2025_1749902667337.pdf` was lost and its article text is used
   instead (53 chunks where the PDF gave 96), so every later chunk ID
   shifted. The manifest records this under `published_reproduction`
   and the verifier warns rather than fails. Never pool nevus retrieval
   distances with pre-migration ones. `chroma_db_nevi/PROVENANCE.md`.
3. ~~Nevus `max_tokens` too low.~~ Now `config.MAX_TOKENS` (8000),
   streamed; a `max_tokens` stop is an error, never a partial parse.
4. ~~Keyword-inference fallback.~~ Gone. Schema enforced server-side; a
   malformed response raises after one retry; the verifier fails any log
   whose `parsing.strategy_used` is not `structured_output` or `failed`.
5. **One preprocessing path.** Never add a second.
6. Real retrieval is noisy (a reference list ranks first for one nevus
   subquery; an affiliation block for one CSCC subquery). Known.
7. Field selection: `extract_tiles.py --source auto` is the default and
   is recorded per case. One line in limitations.
8. **Never enable model fallbacks in `claude_transport.py`.** The SDK
   can substitute another model on refusal. It would be invisible in a
   batch unless every log were read. Refusals are raised, logged with
   their category, counted, and excluded from concordance.
9. **`run_tests.py` fills to a target, it does not append.** It counts
   successful current-protocol logs per case and runs only what is
   missing; a case that keeps failing stops after `target + 2` attempts.
   Re-running is always safe. A log from an older protocol is ignored,
   not deleted.
10. **The manifest is a gate.** `run_tests.py` refuses to run if the
    manifest's protocol differs from the code's or the retrieval context
    hash differs from the manifest's. Re-run `make_manifest.py --yes`
    after any change to the prompt, schema, stores or config.

## Tests

`tests/test_smoke.py` (12 unit tests) and `tests/test_e2e.py` (the whole
chain through the real batch runner, verifier, scorer and report
renderer against a scripted API double: one transient failure, one
refusal, one disagreeing replicate). Neither needs a key or the stores.
`ruff check --select E9,F,B,PLE .` is clean. CI runs all three.

## Conventions

- Do not add a shared abstraction over the two pathways' grading logic.
- The API is called in exactly one place, `claude_transport.grade()`.
- Bump `config.PROTOCOL_VERSION` when the prompt scaffold, schema,
  magnification set, model, or request shape changes; then re-run
  `make_manifest.py --yes`.
- Never hand-edit a case log. Re-run; replicates increment.
- A change that touches retrieval or the prompt needs a
  `verify_logging.py` run and the before/after distances noted.
