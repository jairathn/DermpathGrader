"""Central study configuration (protocol v2.0).

Constants only. No grading logic lives here: the two pathways stay
separately implemented, per the standing convention against a shared
abstraction over the grading code.

Study design
------------
350 cases, 50 per stratum:
  Melanocytic (200): MPATH-Dx v2.0 Class I, II, III, IV
  CSCC        (150): well, moderately, poorly differentiated

The melanocytic strata are the MPATH-Dx v2.0 classes themselves, not the
three-tier dysplasia scale. That resolves the problem the three-tier
design had: v2.0 built its four classes by deleting v1.0's standalone
moderate-atypia category, so a "moderate" case had no single correct
class. Sampling the classes directly removes the ambiguity, and the
arithmetic is unchanged - 4 classes x 50 is the same 200 cases.

The three-tier dysplasia grade is still captured, as a secondary
descriptive field, so mild/moderate/severe can be cross-tabulated
against class. It is no longer what the study is scored on.

Human readers grade whole-slide `.svs`. The model arm grades four JPEG
derivatives per case (whole slide, 4x, 10x, 40x), because the vision API
does not accept `.svs`. docs/STUDY_DESIGN.md covers why that asymmetry
matters and how the analysis handles it.
"""

from __future__ import annotations

# ── Model ────────────────────────────────────────────────────────────
# Both pathways, UI and batch. A mixed-model run is not analysable as one
# experiment, so this is the single point of control.
MODEL_ID = "claude-opus-5"

# Adaptive thinking is on by default on this model; `effort` sets depth.
# Grading is intelligence-sensitive, so this stays high.
THINKING = {"type": "adaptive"}
EFFORT = "high"

# The old call used max_tokens=1500 against an 8-field schema and could
# truncate. The v2 nevus schema is larger (diagnosis, dysplasia grade,
# melanoma subtype, Breslow, MPATH-Dx class, per-magnification evidence).
# Do not lower this.
MAX_TOKENS = 8000

# Streamed, so four base64 images plus a large max_tokens cannot trip the
# HTTP timeout.
STREAM = True

# Sampling parameters were removed on this model family; the old
# temperature=0.1 would now be rejected. Determinism comes from effort
# plus a fixed prompt, and replicates measure what is left.
TEMPERATURE = None

# List price per million tokens for MODEL_ID, used only for the dry-run
# estimate and the post-batch actual. Cache reads are billed at 10% of
# input, which is why the scaffold lives in a cached system block.
PRICE_USD_PER_MTOK = {
    "input": 5.00,
    "output": 25.00,
    "cache_read": 0.50,
    "cache_write": 6.25,
}
# Rough output size of one grading response, for the estimate only.
EST_OUTPUT_TOKENS = 1500

# ── Strata ───────────────────────────────────────────────────────────
N_PER_STRATUM = 50

# MPATH-Dx v2.0 classes. Class 0 (nondiagnostic) is a valid model output
# but is not sampled: a stratum of deliberately ungradeable slides would
# measure scan quality, not grading.
NEVUS_STRATA = ("I", "II", "III", "IV")

# Secondary descriptive fields, captured but not scored as strata.
DYSPLASIA_GRADES = ("mild", "moderate", "severe", "not_applicable")

MELANOMA_SUBTYPES = ("in_situ", "invasive", "not_applicable")

# Histologic subtype, recorded for melanoma cases. Class III and IV split
# on Breslow thickness, not on this, but it is worth capturing: the model
# is asked to name it, and desmoplastic and acral cases are exactly where
# a fixed-field read is expected to struggle.
MELANOMA_HISTOLOGIC_SUBTYPES = (
    "lentigo_maligna",
    "superficial_spreading",
    "nodular",
    "acral_lentiginous",
    "desmoplastic",
    "other",
    "not_applicable",
)
CSCC_STRATA = ("well", "moderately", "poorly")

PATHWAYS = ("CSCC", "Nevus")

STRATA_BY_PATHWAY = {"Nevus": NEVUS_STRATA, "CSCC": CSCC_STRATA}
TARGET_N = {"Nevus": N_PER_STRATUM * len(NEVUS_STRATA),
            "CSCC": N_PER_STRATUM * len(CSCC_STRATA)}
TARGET_N_TOTAL = sum(TARGET_N.values())

# ── Magnifications ───────────────────────────────────────────────────
# Order is the presentation order to the model and the order recorded in
# the logs. Low power first, so architecture is established before
# cytology, mirroring how a slide is actually read.
MAGNIFICATIONS = ("whole_slide", "4x", "10x", "40x")
REQUIRED_MAGNIFICATIONS = MAGNIFICATIONS

MAGNIFICATION_CAPTIONS = {
    "whole_slide": ("whole-slide overview (entire specimen, lowest power) - "
                    "silhouette, symmetry, circumscription, overall size"),
    "4x": ("4x objective - architectural pattern, nest distribution, "
           "confluence, stromal response"),
    "10x": ("10x objective - nest morphology, lateral extension, "
            "maturation with descent, pagetoid spread"),
    "40x": ("40x objective - cytologic atypia, nuclear size relative to "
            "resting basal keratinocytes, nucleoli, mitoses"),
}

# ── Image preprocessing ──────────────────────────────────────────────
# Long-edge cap. Above 1568 px the API downsamples anyway, so extra
# pixels cost upload time and tokens without adding resolution - and at
# four images per case that cost is 4x. The original code had no cap at
# all and sent full-resolution frames.
MAX_IMAGE_EDGE_PX = 1568

# Per-image ceiling on raw bytes. Base64 expands ~4/3, and the API limit
# is 5 MB per image, so 3.5 MB leaves headroom; four of these stay well
# under the 32 MB request cap.
MAX_IMAGE_BYTES = 3_500_000

# Encoder settings. Quality steps down until the byte ceiling is met.
# Histology is unforgiving of hard compression at 40x, so the floor is
# high and the code raises rather than going below it.
JPEG_QUALITY_START = 92
JPEG_QUALITY_FLOOR = 70
JPEG_QUALITY_STEP = 6

OUTBOUND_MEDIA_TYPE = "image/jpeg"
ACCEPTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
WSI_SUFFIXES = (".svs",)

# ── Protocol history ─────────────────────────────────────────────────
#   2.0  four magnifications, MPATH-Dx v2.0 classes as strata, Opus 5,
#        structured outputs, melanoma subtyping
#   2.1  scaffold moved to a cached system block; clinical fields added
#        (adequacy, differential, ancillary studies; CSCC subtype, depth,
#        Broders, high-risk features); transport retries and refusal /
#        truncation recorded per case
#
# ── Retrieval (unchanged from v1 - see CLAUDE.md landmine 2) ─────────
CHROMA_DIR = {"CSCC": "chroma_db", "Nevus": "chroma_db_nevi"}
CHROMA_COLLECTION = {"CSCC": "scc_grading_literature",
                     "Nevus": "nevi_grading_literature"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_PER_SUBQUERY = 5

# ── Paths ────────────────────────────────────────────────────────────
ASSETS_DIR = "attached_assets"
LOG_ROOT = "analysis_logs"
CASE_REGISTRY = "data/case_registry.csv"
READER_MANIFEST = "data/reader_manifest.csv"
TILE_COORDS = "data/tile_coords.csv"
GROUND_TRUTH = {"CSCC": "ground_truth_cscc.csv",
                "Nevus": "ground_truth_nevus.csv"}
RUN_MANIFEST = "run_manifest.json"
TILE_ROOT = "test_images"

# Bump whenever the prompt scaffold, schema, magnification set or model
# changes. Written into every case log, so a mixed-protocol batch is
# detectable after the fact instead of being silently pooled.
PROTOCOL_VERSION = "2.1"
LOG_VERSION = "2.1"
