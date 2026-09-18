"""Batch grading runner (protocol v2.0).

    python run_tests.py [--pathway CSCC|Nevus|both] [--replicates N]
                        [--limit N] [--dry-run]

Reads the case registry, prepares each case's four magnification images,
grades it, and writes one log per case-replicate.

Changes from v1
---------------
- Cases come from data/case_registry.csv, which carries one row per case
  with the .svs path (reader arm) and the four tile paths (model arm).
  v1 read ground truth and guessed at test_images/{pathway}/{filename}.
- Four images per case instead of one, through image_utils, which is the
  same path the Streamlit UI uses.
- The reference grade is still never passed to an analyzer.
- A re-run still always creates a new replicate rather than overwriting.
- The RAG system is still initialised once per pathway so every case in a
  pathway sees a byte-identical context block.
- The st.* monkey-patching v1 needed is gone: the RAG modules now only
  touch Streamlit when a session is actually running.

--dry-run prepares every image and validates the registry without
spending an API call. Run it before committing to a 350-case batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import config
import image_utils
from grading_logger import CaseLogger


def load_manifest() -> dict:
    p = pathlib.Path(config.RUN_MANIFEST)
    if not p.exists():
        sys.exit(f"{config.RUN_MANIFEST} not found. Run make_manifest.py "
                 f"first.")
    return json.loads(p.read_text())


def load_registry(pathway: str) -> list[dict]:
    p = pathlib.Path(config.CASE_REGISTRY)
    if not p.exists():
        sys.exit(f"{p} not found. Run build_case_registry.py first.")
    with p.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("pathway", "").strip() == pathway]
    if not rows:
        sys.exit(f"no {pathway} rows in {p}")
    return rows


def tile_sources(row: dict) -> dict[str, str]:
    """Magnification -> tile path, from a registry row."""
    return {mag: (row.get(f"tile_{mag}", "") or "").strip()
            for mag in config.MAGNIFICATIONS}


def run_case(row: dict, pathway: str, rag_system, manifest: dict,
             dry_run: bool = False) -> pathlib.Path | None:
    case_id = row["case_id"]
    session_id = manifest["session_id"]
    manifest_ctx = (manifest.get("pathways", {}).get(pathway, {})
                    .get("retrieval", {}).get("context_block_sha256", ""))

    prepared = image_utils.prepare_case_images(tile_sources(row))

    if dry_run:
        total_mb = sum(p.sent_bytes for p in prepared) / 1e6
        print(f"  {pathway} {case_id}: {len(prepared)} images, "
              f"{total_mb:.2f} MB")
        return None

    probe = CaseLogger(pathway, case_id, 1, session_id)
    replicate = probe.next_free_replicate()
    logger = CaseLogger(pathway, case_id, replicate, session_id)
    logger._manifest_context_sha256 = manifest_ctx

    logger.set_image_set(
        prepared,
        source_registry=row.get("image_source_registry", ""),
        wsi_source_filename=pathlib.Path(
            row.get("svs_path", "") or "").name,
        wsi_sha256=row.get("svs_sha256", ""),
        tile_selection_method=row.get("tile_selection_method", ""),
    )

    print(f"  {pathway} {case_id} rep{replicate} ... ", end="", flush=True)
    rag_system.clear_retrieval_log()

    # The reference grade is deliberately not passed to the analyzer.
    if pathway == "CSCC":
        from image_analyzer import ImageAnalyzer
        analyzer = ImageAnalyzer(rag_system)
    else:
        from nevi_analyzer import NeviAnalyzer
        analyzer = NeviAnalyzer(rag_system)

    try:
        analyzer.analyze_images(prepared, case_logger=logger)
    except Exception as exc:
        logger.add_error(str(exc))
        path = logger.save()
        print(f"FAILED ({exc}) -> {path}")
        raise

    path = logger.save()
    print(f"done ({logger.record['response']['latency_ms']} ms) -> {path}")
    return path


def build_rag(pathway: str):
    if pathway == "CSCC":
        from rag_system import RAGSystem
        return RAGSystem()
    from nevi_rag_system import NeviRAGSystem
    return NeviRAGSystem()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pathway", choices=["CSCC", "Nevus", "both"],
                    default="both")
    ap.add_argument("--replicates", type=int, default=3,
                    help="replicates per case (minimum 2)")
    ap.add_argument("--limit", type=int,
                    help="only the first N cases per pathway (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare images and validate, spend no API calls")
    args = ap.parse_args()

    if args.replicates < 2:
        sys.exit("--replicates must be at least 2 (spec: minimum two).")
    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set.")

    manifest = load_manifest()
    print(f"Session: {manifest['session_id']}")
    print(f"Protocol: v{config.PROTOCOL_VERSION}  Model: {config.MODEL_ID}")
    print(f"Magnifications: {', '.join(config.MAGNIFICATIONS)}\n")

    pathways = (["CSCC", "Nevus"] if args.pathway == "both"
                else [args.pathway])
    failures = 0

    for pathway in pathways:
        rows = load_registry(pathway)
        if args.limit:
            rows = rows[:args.limit]
        expected = config.TARGET_N[pathway]
        flag = "" if len(rows) == expected else f"  (target is {expected})"
        print(f"-- {pathway}: {len(rows)} cases x {args.replicates} "
              f"replicates{flag} --")

        rag = None if args.dry_run else build_rag(pathway)
        reps = 1 if args.dry_run else args.replicates

        for row in rows:
            for _ in range(reps):
                try:
                    run_case(row, pathway, rag, manifest, args.dry_run)
                except Exception as exc:
                    failures += 1
                    print(f"    error on {row['case_id']}: {exc}")
        print()

    if args.dry_run:
        print("Dry run complete. No API calls were made.")
    else:
        print("Next: python verify_logging.py, then python join_and_score.py")
    if failures:
        print(f"\n{failures} case-replicate(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
