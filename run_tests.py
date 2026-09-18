"""Batch grading runner (protocol v2.1).

    python run_tests.py --dry-run                      validate + cost estimate
    python run_tests.py                                fill to 3 replicates
    python run_tests.py --pathway Nevus --workers 4
    python run_tests.py --case-id NEV-0042             one case

Resumable by construction
-------------------------
The target is N *successful* replicates per case (default 3). On every
run the script counts the current-protocol logs already on disk for each
case and runs only what is missing. Kill it, fix the problem, run it
again: nothing is duplicated and nothing is overwritten. A case that
keeps failing (refusal, truncation) stops after `target + 2` total
attempts rather than looping.

Concurrency
-----------
`--workers N` grades N cases at once. The retrieval context is computed
ONCE per pathway before any worker starts and handed to every analyzer as
a frozen, read-only object. That is both the thread-safety fix (the RAG
systems keep per-call state) and a reproducibility guarantee: every case
in the batch provably saw the same context block, and its hash is checked
against the manifest before the first API call.

Cost
----
`--dry-run` prepares every image, validates the registry, and prints a
token and dollar estimate. With an API key present it uses the real
count_tokens endpoint for the scaffold; without one it estimates. After a
real run, the actual spend is computed from the usage recorded in each
log and written to `analysis_logs/batch_summary.json`.

The reference grade never reaches an analyzer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import pathlib
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field

import config
import credentials
import image_utils
from grading_logger import CaseLogger


# ── frozen retrieval ─────────────────────────────────────────────────

class FrozenRAG:
    """Read-only stand-in for a RAG system, safe to share across threads.

    Holds the context block and retrieval log computed once by the real
    RAG system, and returns them unchanged on every call.
    """

    def __init__(self, context: str, retrieval_log: list[dict]):
        self._context = context
        self._retrieval_log = list(retrieval_log)

    def get_grading_criteria(self) -> str:
        return self._context

    def clear_retrieval_log(self) -> None:
        pass   # nothing to clear: the log is fixed for the whole batch


def freeze_retrieval(pathway: str, manifest: dict) -> FrozenRAG:
    """Run retrieval once, check it against the manifest, freeze it."""
    if pathway == "CSCC":
        from rag_system import RAGSystem
        rag = RAGSystem()
    else:
        from nevi_rag_system import NeviRAGSystem
        rag = NeviRAGSystem()
    rag.clear_retrieval_log()
    context = rag.get_grading_criteria()
    observed = hashlib.sha256(context.encode()).hexdigest()
    expected = (manifest.get("pathways", {}).get(pathway, {})
                .get("retrieval", {}).get("context_block_sha256", ""))
    if expected and observed != expected:
        sys.exit(
            f"{pathway}: retrieval context does not match run_manifest.json\n"
            f"  manifest {expected}\n  observed {observed}\n"
            f"The store or the subquery panel changed since the manifest was "
            f"built. Re-run make_manifest.py, or restore the store, before "
            f"grading: a batch graded against an unmanifested context is not "
            f"reproducible.")
    return FrozenRAG(context, rag.get_retrieval_log())


# ── registry and logs ────────────────────────────────────────────────

def load_manifest() -> dict:
    p = pathlib.Path(config.RUN_MANIFEST)
    if not p.exists():
        sys.exit(f"{config.RUN_MANIFEST} not found. Run make_manifest.py.")
    m = json.loads(p.read_text())
    if m.get("protocol_version") != config.PROTOCOL_VERSION:
        sys.exit(f"run_manifest.json is protocol {m.get('protocol_version')!r} "
                 f"but the code is {config.PROTOCOL_VERSION!r}. Re-run "
                 f"make_manifest.py so the manifest describes what will run.")
    return m


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
    return {mag: (row.get(f"tile_{mag}", "") or "").strip()
            for mag in config.MAGNIFICATIONS}


@dataclass
class CaseState:
    successful: int = 0
    failed: int = 0
    stale: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.successful + self.failed


def existing_state(pathway: str, case_id: str) -> CaseState:
    """What is already on disk for this case, current protocol only."""
    state = CaseState()
    directory = pathlib.Path(config.LOG_ROOT) / pathway
    for path in sorted(directory.glob(f"{case_id}__rep*.json")):
        try:
            log = json.loads(path.read_text())
        except json.JSONDecodeError:
            state.failed += 1
            state.failures.append(f"{path.name}: unreadable")
            continue
        if log.get("protocol_version") != config.PROTOCOL_VERSION:
            state.stale += 1
            continue
        if log.get("failure"):
            state.failed += 1
            state.failures.append(
                f"{path.name}: {log['failure'].get('error_class')}")
        else:
            state.successful += 1
    return state


# ── cost ─────────────────────────────────────────────────────────────

def estimate_tokens(prepared: list, system_text: str, user_text: str,
                    client=None) -> dict[str, int]:
    """Input tokens for one call, split into cacheable and per-case."""
    if client is not None:
        try:
            count = client.messages.count_tokens(
                model=config.MODEL_ID,
                system=[{"type": "text", "text": system_text}],
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": user_text}]}])
            scaffold = int(count.input_tokens)
        except Exception:
            scaffold = len(system_text) // 4
    else:
        scaffold = len(system_text) // 4
    # ~750 pixels per token is the documented rule of thumb for images.
    images = sum((p.sent_dimensions_px[0] * p.sent_dimensions_px[1]) // 750
                 for p in prepared)
    captions = sum(len(config.MAGNIFICATION_CAPTIONS[p.magnification]) // 4
                   for p in prepared)
    return {"scaffold": scaffold, "per_case": images + captions
            + len(user_text) // 4}


def estimate_cost_usd(n_calls: int, tokens: dict[str, int]) -> dict:
    price = config.PRICE_USD_PER_MTOK
    # First call writes the cache; the rest read it (within the cache
    # window - a slow batch may re-write occasionally, so this is a floor).
    scaffold_cost = (tokens["scaffold"] * price["cache_write"]
                     + tokens["scaffold"] * price["cache_read"] * (n_calls - 1)
                     ) / 1e6
    per_case_cost = tokens["per_case"] * n_calls * price["input"] / 1e6
    output_cost = config.EST_OUTPUT_TOKENS * n_calls * price["output"] / 1e6
    uncached = (tokens["scaffold"] + tokens["per_case"]) * n_calls * price["input"] / 1e6 + output_cost
    return {
        "calls": n_calls,
        "scaffold_tokens_per_call": tokens["scaffold"],
        "per_case_input_tokens": tokens["per_case"],
        "est_output_tokens_per_call": config.EST_OUTPUT_TOKENS,
        "usd_with_cache": round(scaffold_cost + per_case_cost + output_cost, 2),
        "usd_without_cache": round(uncached, 2),
    }


def actual_cost_usd(logs: list[dict]) -> dict:
    price = config.PRICE_USD_PER_MTOK
    totals = Counter()
    for log in logs:
        u = log.get("response", {}).get("usage", {})
        cache_read = u.get("cache_read_input_tokens", 0) or 0
        cache_write = u.get("cache_creation_input_tokens", 0) or 0
        plain = max(0, (u.get("input_tokens", 0) or 0))
        totals["input"] += plain
        totals["cache_read"] += cache_read
        totals["cache_write"] += cache_write
        totals["output"] += u.get("output_tokens", 0) or 0
    usd = sum(totals[k] * price[k] for k in totals) / 1e6
    return {"tokens": dict(totals), "usd": round(usd, 2),
            "cache_hit_calls": sum(
                1 for l in logs
                if (l.get("response", {}).get("usage", {})
                    .get("cache_read_input_tokens", 0) or 0) > 0)}


# ── one case ─────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _say(text: str) -> None:
    with _print_lock:
        print(text, flush=True)


def run_case(row: dict, pathway: str, frozen: FrozenRAG, manifest: dict,
             client=None) -> pathlib.Path:
    """Grade one replicate of one case. Always leaves a log behind."""
    case_id = row["case_id"]
    session_id = manifest["session_id"]
    manifest_ctx = (manifest.get("pathways", {}).get(pathway, {})
                    .get("retrieval", {}).get("context_block_sha256", ""))

    prepared = image_utils.prepare_case_images(tile_sources(row))

    probe = CaseLogger(pathway, case_id, 1, session_id)
    replicate = probe.next_free_replicate()
    logger = CaseLogger(pathway, case_id, replicate, session_id)
    logger._manifest_context_sha256 = manifest_ctx
    logger.set_image_set(
        prepared,
        source_registry=row.get("image_source_registry", ""),
        wsi_source_filename=pathlib.Path(row.get("svs_path", "") or "").name,
        wsi_sha256=row.get("svs_sha256", ""),
        tile_selection_method=row.get("tile_selection_method", ""),
    )

    if pathway == "CSCC":
        from image_analyzer import ImageAnalyzer
        analyzer = ImageAnalyzer(frozen, client=client)
    else:
        from nevi_analyzer import NeviAnalyzer
        analyzer = NeviAnalyzer(frozen, client=client)

    try:
        analyzer.analyze_images(prepared, case_logger=logger)
    except Exception as exc:
        if logger.record.get("failure") is None:
            logger.set_failure(exc)
        path = logger.save()
        _say(f"  {pathway} {case_id} rep{replicate} FAILED "
             f"{type(exc).__name__}: {str(exc)[:80]} -> {path.name}")
        raise
    path = logger.save()
    _say(f"  {pathway} {case_id} rep{replicate} ok "
         f"({logger.record['response']['latency_ms']} ms, "
         f"attempt {logger.record['request']['attempt']}) -> {path.name}")
    return path


# ── batch ────────────────────────────────────────────────────────────

def plan_pathway(rows: list[dict], pathway: str, target: int
                 ) -> tuple[list[tuple[dict, int]], dict]:
    """(case, replicates_to_run) for every case still short of target."""
    work: list[tuple[dict, int]] = []
    summary = Counter()
    for row in rows:
        state = existing_state(pathway, row["case_id"])
        summary["cases"] += 1
        summary["done_replicates"] += state.successful
        summary["failed_replicates"] += state.failed
        summary["stale_logs"] += state.stale
        if state.successful >= target:
            summary["complete_cases"] += 1
            continue
        if state.total >= target + 2:
            summary["given_up"] += 1
            _say(f"  {pathway} {row['case_id']}: {state.successful}/{target} "
                 f"after {state.total} attempts; giving up. "
                 f"{'; '.join(state.failures[-2:])}")
            continue
        need = target - state.successful
        work.append((row, need))
        summary["planned_replicates"] += need
    return work, dict(summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pathway", choices=["CSCC", "Nevus", "both"], default="both")
    ap.add_argument("--target-replicates", type=int, default=3,
                    help="successful replicates wanted per case (min 2)")
    ap.add_argument("--workers", type=int, default=2,
                    help="cases graded concurrently")
    ap.add_argument("--limit", type=int, help="first N cases per pathway")
    ap.add_argument("--case-id", action="append", default=[],
                    help="restrict to these case ids (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare images, validate, estimate cost; no API calls")
    args = ap.parse_args()

    if args.target_replicates < 2:
        sys.exit("--target-replicates must be at least 2.")
    if args.workers < 1:
        sys.exit("--workers must be at least 1.")
    have_key = bool(credentials.ensure_api_key())
    if not args.dry_run and not have_key:
        sys.exit(
            "ANTHROPIC_API_KEY not found in the environment or a .env file "
            "in the project root.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...   or put it in .env\n"
            "Run `python doctor.py` to check everything at once.")

    manifest = load_manifest()
    print(f"Session   {manifest['session_id']}")
    print(f"Protocol  v{config.PROTOCOL_VERSION}   Model {config.MODEL_ID}   "
          f"effort {config.EFFORT}")
    print(f"Images    {', '.join(config.MAGNIFICATIONS)}")
    print(f"Target    {args.target_replicates} successful replicates per case, "
          f"{args.workers} worker(s)\n")

    client = None
    if have_key:
        from anthropic import Anthropic
        client = Anthropic()

    pathways = ["CSCC", "Nevus"] if args.pathway == "both" else [args.pathway]
    overall = {"started_utc": datetime.datetime.now(datetime.timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%SZ"), "pathways": {}}
    any_failures = False

    for pathway in pathways:
        rows = load_registry(pathway)
        if args.case_id:
            rows = [r for r in rows if r["case_id"] in args.case_id]
        if args.limit:
            rows = rows[:args.limit]
        expected = config.TARGET_N[pathway]
        flag = "" if len(rows) == expected else f"  (design target {expected})"
        print(f"-- {pathway}: {len(rows)} cases{flag} --")

        work, summary = plan_pathway(rows, pathway, args.target_replicates)
        print(f"  on disk: {summary.get('done_replicates', 0)} successful, "
              f"{summary.get('failed_replicates', 0)} failed, "
              f"{summary.get('stale_logs', 0)} stale-protocol; "
              f"{summary.get('complete_cases', 0)} case(s) complete; "
              f"planned {summary.get('planned_replicates', 0)} replicate(s)")

        if args.dry_run:
            problems = 0
            first_prepared = None
            for row, _ in work:
                try:
                    prepared = image_utils.prepare_case_images(tile_sources(row))
                    first_prepared = first_prepared or prepared
                except Exception as exc:
                    problems += 1
                    print(f"    {row['case_id']}: {exc}")
            if first_prepared is not None:
                if pathway == "CSCC":
                    from image_analyzer import ImageAnalyzer as A
                    import image_analyzer as mod
                else:
                    from nevi_analyzer import NeviAnalyzer as A
                    import nevi_analyzer as mod
                system_text = A.build_system_prompt("x" * 45000)
                tokens = estimate_tokens(first_prepared, system_text,
                                         mod.USER_INSTRUCTION, client)
                n_calls = summary.get("planned_replicates", 0)
                est = estimate_cost_usd(n_calls, tokens)
                print(f"  estimate for {n_calls} call(s): "
                      f"${est['usd_with_cache']} with scaffold caching "
                      f"(${est['usd_without_cache']} without); "
                      f"{tokens['scaffold']} scaffold + {tokens['per_case']} "
                      f"per-case input tokens per call")
                overall["pathways"][pathway] = {"dry_run": True,
                                                "estimate": est, **summary}
            print(f"  {len(work) - problems}/{len(work)} planned case(s) "
                  f"prepare cleanly\n")
            any_failures |= problems > 0
            continue

        frozen = freeze_retrieval(pathway, manifest)
        jobs = [(row, pathway) for row, need in work for _ in range(need)]
        ok = failed = 0
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, row, pw, frozen, manifest, client)
                       for row, pw in jobs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    ok += 1
                except Exception:
                    failed += 1

        logs = []
        for row in rows:
            for path in (pathlib.Path(config.LOG_ROOT) / pathway).glob(
                    f"{row['case_id']}__rep*.json"):
                log = json.loads(path.read_text())
                if (log.get("protocol_version") == config.PROTOCOL_VERSION
                        and not log.get("failure")):
                    logs.append(log)
        cost = actual_cost_usd(logs)
        print(f"  ran {ok + failed}: {ok} ok, {failed} failed. "
              f"Actual spend to date this protocol: ${cost['usd']} over "
              f"{len(logs)} successful log(s); cache hit on "
              f"{cost['cache_hit_calls']}.\n")
        overall["pathways"][pathway] = {"ran": ok + failed, "ok": ok,
                                        "failed": failed, "cost": cost,
                                        **summary}
        any_failures |= failed > 0

    overall["finished_utc"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = pathlib.Path(config.LOG_ROOT) / "batch_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overall, indent=2))
    print(f"Summary -> {out}")
    if args.dry_run:
        print("Dry run: no API calls were made.")
    else:
        print("Next: python verify_logging.py, then python join_and_score.py")
        print("Re-run this command to fill any case still short of target.")
    if any_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
