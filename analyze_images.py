"""
analyze_images.py
-----------------
CLI runner: analyzes one or more images through the full RAG+LLM pipeline
and writes a structured JSON log to analysis_logs/{pathway}/.

Usage:
    python analyze_images.py --scc   attached_assets/SCC_1785341664948.jpeg
    python analyze_images.py --nevus attached_assets/Nevus_1785341664947.jpg
    python analyze_images.py --scc <file> --nevus <file>   # both at once
"""

import argparse
import base64
import io
import json
import pathlib
import sys

from PIL import Image

from grading_logger import CaseLogger, sha256_bytes, sha256_str
import image_utils


# ── manifest session_id ───────────────────────────────────────────────────────

def _session_id() -> str:
    try:
        return json.loads(pathlib.Path("run_manifest.json").read_text())["session_id"]
    except Exception:
        return "ui-no-manifest"


def _manifest_context_sha(pathway: str) -> str:
    try:
        m = json.loads(pathlib.Path("run_manifest.json").read_text())
        key = "CSCC" if pathway == "CSCC" else "Nevus"
        return m["pathways"][key]["retrieval"]["context_block_sha256"]
    except Exception:
        return ""


# ── image preparation ─────────────────────────────────────────────────────────

def prepare_image(path: pathlib.Path):
    """Return (base64_str, media_type, meta_dict) for a single image.

    Delegates to image_utils so this reporting script cannot drift away
    from the preprocessing the grading run actually used. `magnification`
    is passed as "unspecified" because this helper handles one loose
    image; a real protocol v2.0 case has four, and is prepared with
    image_utils.prepare_case_images().
    """
    prepared = image_utils.prepare_image(path, "unspecified")
    meta = prepared.to_log_dict()
    meta.pop("magnification", None)
    return prepared.b64, prepared.sent_media_type, meta


# ── next free replicate ───────────────────────────────────────────────────────

def next_rep(pathway: str, case_id: str) -> int:
    probe = CaseLogger(pathway, case_id, 1, "probe")
    return probe.next_free_replicate()


# ── SCC analysis ──────────────────────────────────────────────────────────────

def run_scc(image_path: pathlib.Path, session_id: str):
    print(f"\n[SCC] Loading image: {image_path}")
    b64, media_type, meta = prepare_image(image_path)

    print("[SCC] Initializing RAG system…")
    from rag_system import RAGSystem
    from image_analyzer import ImageAnalyzer
    rag = RAGSystem()

    stem = image_path.stem.replace(" ", "_")
    case_id = f"UI-{stem}"
    rep = next_rep("CSCC", case_id)
    logger = CaseLogger("CSCC", case_id, rep, session_id)

    logger.set_image_meta(
        source_registry="ui_upload",
        source_filename=meta["source_filename"],
        source_sha256=meta["source_sha256"],
        source_dimensions_px=meta["source_dimensions_px"],
        sent_media_type=meta["sent_media_type"],
        sent_dimensions_px=meta["sent_dimensions_px"],
        sent_bytes=meta["sent_bytes"],
        sent_sha256=meta["sent_sha256"],
        resize_applied=meta["resize_applied"],
        compression_quality=meta["compression_quality"],
    )

    # Supply the manifest context SHA so the analyzer can set matches_manifest_context
    logger._manifest_context_sha256 = _manifest_context_sha("CSCC")

    print("[SCC] Running analysis (Claude API call)…")
    analyzer = ImageAnalyzer(rag)
    # analyze_image() calls logger.set_retrieval() internally using _manifest_context_sha256
    result = analyzer.analyze_image(b64, media_type, case_logger=logger)

    fp = logger.save()
    print(f"[SCC] Log saved → {fp}")
    return result, fp


# ── Nevus analysis ────────────────────────────────────────────────────────────

def run_nevus(image_path: pathlib.Path, session_id: str):
    print(f"\n[Nevus] Loading image: {image_path}")
    b64, media_type, meta = prepare_image(image_path)

    print("[Nevus] Initializing RAG system…")
    from nevi_rag_system import NeviRAGSystem
    from nevi_analyzer import NeviAnalyzer
    rag = NeviRAGSystem()

    stem = image_path.stem.replace(" ", "_")
    case_id = f"UI-{stem}"
    rep = next_rep("Nevus", case_id)
    logger = CaseLogger("Nevus", case_id, rep, session_id)

    logger.set_image_meta(
        source_registry="ui_upload",
        source_filename=meta["source_filename"],
        source_sha256=meta["source_sha256"],
        source_dimensions_px=meta["source_dimensions_px"],
        sent_media_type=meta["sent_media_type"],
        sent_dimensions_px=meta["sent_dimensions_px"],
        sent_bytes=meta["sent_bytes"],
        sent_sha256=meta["sent_sha256"],
        resize_applied=meta["resize_applied"],
        compression_quality=meta["compression_quality"],
    )

    # Supply the manifest context SHA so the analyzer can set matches_manifest_context
    logger._manifest_context_sha256 = _manifest_context_sha("Nevus")

    print("[Nevus] Running analysis (Claude API call)…")
    analyzer = NeviAnalyzer(rag)
    # analyze_image() calls logger.set_retrieval() internally using _manifest_context_sha256
    result = analyzer.analyze_image(b64, media_type, case_logger=logger)

    fp = logger.save()
    print(f"[Nevus] Log saved → {fp}")
    return result, fp


# ── pretty-print ──────────────────────────────────────────────────────────────

def print_scc_report(result: dict, log_path: pathlib.Path):
    print("\n" + "═" * 60)
    print("  SCC GRADING REPORT")
    print("═" * 60)
    print(f"  Grade:              {result.get('primary_grade', 'N/A')}")
    print(f"  Confidence:         {result.get('confidence_level', 'N/A')}")
    print(f"  Keratinization:     {result.get('keratinization_present', 'N/A')}")
    print(f"  Atypia level:       {result.get('atypia_level', 'N/A')}")
    feats = result.get("key_features", [])
    if feats:
        print("  Key features:")
        for f in feats:
            print(f"    • {f}")
    obs = result.get("additional_observations", "")
    if obs:
        print(f"  Additional observations:\n    {obs}")
    print(f"\n  Full log: {log_path}")
    print("═" * 60)


def print_nevus_report(result: dict, log_path: pathlib.Path):
    print("\n" + "═" * 60)
    print("  NEVUS GRADING REPORT")
    print("═" * 60)
    print(f"  Traditional grade:  {result.get('traditional_grade', 'N/A')}")
    print(f"  MPATH-Dx grade:     {result.get('mpath_grade', 'N/A')}")
    print(f"  Confidence:         {result.get('confidence_level', 'N/A')}")
    print(f"  Nuclear abnorm.:    {result.get('nuclear_abnormality_count', 'N/A')}")
    arch = result.get("architectural_features", [])
    if arch:
        print("  Architectural features:")
        for f in arch:
            print(f"    • {f}")
    cyto = result.get("cytological_features", [])
    if cyto:
        print("  Cytological features:")
        for f in cyto:
            print(f"    • {f}")
    obs = result.get("additional_observations", "")
    if obs:
        print(f"  Grading rationale:\n    {obs}")
    clin = result.get("clinical_significance", "")
    if clin:
        print(f"  Clinical significance:\n    {clin}")
    print(f"\n  Full log: {log_path}")
    print("═" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scc",   metavar="FILE", help="Path to SCC image")
    ap.add_argument("--nevus", metavar="FILE", help="Path to Nevus image")
    args = ap.parse_args()

    if not args.scc and not args.nevus:
        ap.error("Provide at least one of --scc or --nevus")

    sid = _session_id()

    if args.scc:
        result, fp = run_scc(pathlib.Path(args.scc), sid)
        print_scc_report(result, fp)

    if args.nevus:
        result, fp = run_nevus(pathlib.Path(args.nevus), sid)
        print_nevus_report(result, fp)


if __name__ == "__main__":
    main()
