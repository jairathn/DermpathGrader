"""
run_tests.py
------------
Step 4 of the logging workflow.  Run after make_manifest.py and after
filling the ground truth CSVs.

    python run_tests.py [--pathway CSCC|Nevus|both] [--replicates N]

Images must be placed in:
    test_images/CSCC/<source_filename>
    test_images/Nevus/<source_filename>

where source_filename comes from the ground truth CSV.

Constraints enforced here:
  • Analysis modules never receive the reference grade (constraint 3).
  • A re-run always creates a new replicate file (constraint 5).
  • The RAG system is initialised once and reused across all cases (so the
    context block hash is identical for every case in a pathway).
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import pathlib
import sys
import time
import datetime

from PIL import Image

from grading_logger import CaseLogger, sha256_bytes, sha256_str

# ── helpers ──────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    p = pathlib.Path("run_manifest.json")
    if not p.exists():
        sys.exit("run_manifest.json not found.  Run make_manifest.py first.")
    return json.loads(p.read_text())


def load_ground_truth(pathway: str) -> list[dict]:
    fname = "ground_truth_cscc.csv" if pathway == "CSCC" else "ground_truth_nevus.csv"
    p = pathlib.Path(fname)
    if not p.exists():
        sys.exit(f"{fname} not found.  Create it with the required columns.")
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def find_image(pathway: str, source_filename: str) -> pathlib.Path:
    p = pathlib.Path("test_images") / pathway / source_filename
    if not p.exists():
        raise FileNotFoundError(
            f"Image not found: {p}\n"
            f"Place images in test_images/{pathway}/"
        )
    return p


def process_image_for_api(img_path: pathlib.Path):
    """
    Apply the same pipeline as image_utils.validate_and_process_image but
    from a file path (no Streamlit dependency).  Returns:
        image_data (base64 str), media_type, image_meta (dict)
    """
    from image_utils import compress_image_for_api, image_to_base64, get_last_image_meta

    # Source bytes and SHA-256
    raw_bytes  = img_path.read_bytes()
    source_sha = sha256_bytes(raw_bytes)

    image = Image.open(io.BytesIO(raw_bytes))
    source_dims = list(image.size)  # (width, height)

    processed_image, was_compressed = compress_image_for_api(image)
    sent_dims = list(processed_image.size)

    image_data, media_type = image_to_base64(processed_image, was_compressed)

    # Sent bytes = actual bytes the model receives (base64-decoded)
    sent_raw   = base64.b64decode(image_data)
    sent_bytes = len(sent_raw)
    sent_sha   = sha256_bytes(sent_raw)

    image_meta = {
        "source_sha256":        source_sha,
        "source_dimensions_px": source_dims,
        "sent_media_type":      media_type,
        "sent_dimensions_px":   sent_dims,
        "sent_bytes":           sent_bytes,
        "sent_sha256":          sent_sha,
        "resize_applied":       was_compressed,
        "compression_quality":  None,   # quality loop internals not exposed
    }

    return image_data, media_type, image_meta


# ── CSCC pathway ──────────────────────────────────────────────────────────────

def run_cscc_case(case_row: dict, replicate: int,
                  rag_system, manifest: dict) -> pathlib.Path:
    case_id = case_row["case_id"]
    pathway = "CSCC"

    session_id  = manifest["session_id"]
    ctx_sha_manifest = manifest["pathways"]["CSCC"]["retrieval"]["context_block_sha256"]

    logger = CaseLogger(pathway, case_id, replicate, session_id)
    logger._manifest_context_sha256 = ctx_sha_manifest

    # Resolve next free replicate so we never overwrite
    rep = logger.next_free_replicate()
    logger = CaseLogger(pathway, case_id, rep, session_id)
    logger._manifest_context_sha256 = ctx_sha_manifest

    print(f"  {pathway} {case_id} rep{rep} … ", end="", flush=True)

    # Image
    img_path = find_image(pathway, case_row["source_filename"])
    image_data, media_type, img_meta = process_image_for_api(img_path)

    logger.set_image_meta(
        source_registry    = case_row.get("image_source_registry", ""),
        source_filename    = case_row["source_filename"],
        source_sha256      = img_meta["source_sha256"],
        source_dimensions_px = img_meta["source_dimensions_px"],
        sent_media_type    = img_meta["sent_media_type"],
        sent_dimensions_px = img_meta["sent_dimensions_px"],
        sent_bytes         = img_meta["sent_bytes"],
        sent_sha256        = img_meta["sent_sha256"],
        resize_applied     = img_meta["resize_applied"],
        compression_quality = img_meta["compression_quality"],
    )

    # Clear retrieval log before this case (reusing shared rag_system)
    rag_system.clear_retrieval_log()

    # Analysis (reference grade is NOT passed to the analyzer)
    from image_analyzer import ImageAnalyzer
    analyzer = ImageAnalyzer(rag_system)
    analyzer.analyze_image(image_data, media_type, case_logger=logger)

    fp = logger.save()
    elapsed = logger.record["response"]["latency_ms"]
    print(f"done ({elapsed} ms)  → {fp}")
    return fp


# ── Nevus pathway ──────────────────────────────────────────────────────────────

def run_nevus_case(case_row: dict, replicate: int,
                   rag_system, manifest: dict) -> pathlib.Path:
    case_id = case_row["case_id"]
    pathway = "Nevus"

    session_id           = manifest["session_id"]
    ctx_sha_manifest     = manifest["pathways"]["Nevus"]["retrieval"]["context_block_sha256"]

    logger = CaseLogger(pathway, case_id, replicate, session_id)
    logger._manifest_context_sha256 = ctx_sha_manifest

    rep = logger.next_free_replicate()
    logger = CaseLogger(pathway, case_id, rep, session_id)
    logger._manifest_context_sha256 = ctx_sha_manifest

    print(f"  {pathway} {case_id} rep{rep} … ", end="", flush=True)

    img_path = find_image(pathway, case_row["source_filename"])
    image_data, media_type, img_meta = process_image_for_api(img_path)

    logger.set_image_meta(
        source_registry    = case_row.get("image_source_registry", ""),
        source_filename    = case_row["source_filename"],
        source_sha256      = img_meta["source_sha256"],
        source_dimensions_px = img_meta["source_dimensions_px"],
        sent_media_type    = img_meta["sent_media_type"],
        sent_dimensions_px = img_meta["sent_dimensions_px"],
        sent_bytes         = img_meta["sent_bytes"],
        sent_sha256        = img_meta["sent_sha256"],
        resize_applied     = img_meta["resize_applied"],
        compression_quality = img_meta["compression_quality"],
    )

    rag_system.clear_retrieval_log()

    from nevi_analyzer import NeviAnalyzer
    analyzer = NeviAnalyzer(rag_system)
    analyzer.analyze_image(image_data, media_type, case_logger=logger)

    fp = logger.save()
    elapsed = logger.record["response"]["latency_ms"]
    print(f"done ({elapsed} ms)  → {fp}")
    return fp


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathway",    choices=["CSCC", "Nevus", "both"],
                        default="both")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Number of replicates per case (min 2, default 3)")
    args = parser.parse_args()

    if args.replicates < 2:
        sys.exit("--replicates must be at least 2 (spec: minimum two replicates).")

    api_key = __import__("os").getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set.")

    manifest = load_manifest()
    print(f"Session: {manifest['session_id']}\n")

    run_pathways = (["CSCC", "Nevus"] if args.pathway == "both"
                    else [args.pathway])

    for pathway in run_pathways:
        cases = load_ground_truth(pathway)
        print(f"── {pathway}: {len(cases)} cases × {args.replicates} replicates ──")

        # Initialise RAG system once for the pathway (do NOT re-embed)
        if pathway == "CSCC":
            # Suppress Streamlit calls inside RAGSystem by monkey-patching st
            import streamlit as st
            _orig_info    = st.info
            _orig_success = st.success
            _orig_warning = st.warning
            _orig_error   = st.error
            st.info    = lambda *a, **k: None
            st.success = lambda *a, **k: None
            st.warning = lambda *a, **k: None
            st.error   = lambda *a, **k: None
            try:
                from rag_system import RAGSystem
                rag = RAGSystem()
            finally:
                st.info    = _orig_info
                st.success = _orig_success
                st.warning = _orig_warning
                st.error   = _orig_error
        else:
            import streamlit as st
            _orig_info    = st.info
            _orig_success = st.success
            _orig_warning = st.warning
            _orig_error   = st.error
            st.info    = lambda *a, **k: None
            st.success = lambda *a, **k: None
            st.warning = lambda *a, **k: None
            st.error   = lambda *a, **k: None
            try:
                from nevi_rag_system import NeviRAGSystem
                rag = NeviRAGSystem()
            finally:
                st.info    = _orig_info
                st.success = _orig_success
                st.warning = _orig_warning
                st.error   = _orig_error

        for case_row in cases:
            for rep in range(1, args.replicates + 1):
                try:
                    if pathway == "CSCC":
                        run_cscc_case(case_row, rep, rag, manifest)
                    else:
                        run_nevus_case(case_row, rep, rag, manifest)
                except FileNotFoundError as e:
                    print(f"\n  ⚠  Skipped: {e}\n")
                except Exception as e:
                    print(f"\n  ❌  Error on {case_row['case_id']} rep{rep}: {e}\n")

        print()

    print("All done.  Next: python join_and_score.py")


if __name__ == "__main__":
    main()
