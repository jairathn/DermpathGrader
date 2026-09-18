"""Preflight: is this machine ready to grade?

    python doctor.py            offline checks
    python doctor.py --ping     also confirm the API key can reach the model

Run it before a batch and before telling a researcher the app is up. It
exits non-zero on anything that would make a run fail or make its output
untrustworthy, and it says which.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import pathlib
import sys
from collections import Counter

import config
import credentials

_results: list[tuple[str, str, str]] = []   # (level, name, detail)


def ok(name: str, detail: str = "") -> None:
    _results.append(("ok", name, detail))


def warn(name: str, detail: str) -> None:
    _results.append(("warn", name, detail))


def fail(name: str, detail: str) -> None:
    _results.append(("FAIL", name, detail))


def check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        ok("python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        fail("python", f"{v.major}.{v.minor} found; 3.11+ required")


def check_imports() -> None:
    for name, _minimum in (("anthropic", "1.0.0"), ("chromadb", "1.0.0"),
                          ("PIL", "11.0.0"), ("numpy", "1.26")):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            ok(f"import {name}", version)
            if name == "anthropic" and version != "?" and version.split(".")[0] == "0":
                fail("anthropic version",
                     f"{version} is 0.x; structured outputs and adaptive "
                     "thinking need 1.x (pip install -U anthropic)")
        except ImportError as exc:
            fail(f"import {name}", str(exc))
    try:
        importlib.import_module("streamlit")
        ok("import streamlit")
    except ImportError:
        warn("import streamlit", "not installed; app.py will not run "
             "(batch tools do not need it)")
    try:
        importlib.import_module("openslide")
        ok("import openslide", "extract_tiles.py can read .svs")
    except ImportError:
        warn("import openslide", "not installed; extract_tiles.py needs it "
             "unless tiles come from ImageScope/QuPath")


def check_sqlite() -> None:
    import sqlite_compat
    status = sqlite_compat.STATUS
    if "will fail" in status:
        fail("sqlite", status)
    else:
        ok("sqlite", status)


def check_stores(manifest: dict | None) -> None:
    try:
        import sqlite_compat  # noqa: F401
        import chromadb
    except ImportError:
        return
    for pathway in config.PATHWAYS:
        directory = pathlib.Path(config.CHROMA_DIR[pathway])
        if not directory.exists() or not any(directory.iterdir()):
            fail(f"store {pathway}", f"{directory}/ absent; run "
                 f"build_vector_stores.py --pathway {pathway}")
            continue
        try:
            col = chromadb.PersistentClient(path=str(directory)).get_collection(
                config.CHROMA_COLLECTION[pathway])
            count = col.count()
        except Exception as exc:
            fail(f"store {pathway}", f"cannot open: {exc}")
            continue
        expected = None
        if manifest:
            expected = (manifest.get("pathways", {}).get(pathway, {})
                        .get("vector_store", {}).get("chunk_count"))
        if expected is not None and count != expected:
            fail(f"store {pathway}", f"{count} chunks on disk, manifest says "
                 f"{expected}; re-run make_manifest.py or restore the store")
        else:
            ok(f"store {pathway}", f"{count} chunks")


def check_manifest() -> dict | None:
    p = pathlib.Path(config.RUN_MANIFEST)
    if not p.exists():
        fail("manifest", f"{p} missing; run make_manifest.py --yes")
        return None
    try:
        m = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        fail("manifest", f"unreadable: {exc}")
        return None
    if m.get("protocol_version") != config.PROTOCOL_VERSION:
        fail("manifest", f"protocol {m.get('protocol_version')!r} but code is "
             f"{config.PROTOCOL_VERSION!r}; re-run make_manifest.py --yes")
    else:
        ok("manifest", f"session {m.get('session_id')}, protocol "
           f"{m.get('protocol_version')}")
    for pathway in config.PATHWAYS:
        rep = m.get("pathways", {}).get(pathway, {}).get("published_reproduction")
        if rep and not rep.get("reproduced"):
            warn(f"manifest {pathway}", "store diverges from pre-migration "
                 "published values (recorded and accepted); do not pool "
                 "retrieval distances with old ones")
    return m


def check_key(ping: bool) -> None:
    credentials.ensure_api_key()
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        fail("api key",
             "ANTHROPIC_API_KEY not found in the environment, Streamlit "
             "secrets, or a .env file in the project root. Never commit it "
             "to the repository.")
        return
    ok("api key", f"present ({key[:7]}...{key[-4:]}) via "
       f"{credentials.describe_source()}")
    if not ping:
        return
    try:
        from anthropic import Anthropic
        client = Anthropic()
        count = client.messages.count_tokens(
            model=config.MODEL_ID,
            messages=[{"role": "user", "content": "ping"}])
        ok("api ping", f"{config.MODEL_ID} reachable "
           f"(count_tokens -> {count.input_tokens})")
    except Exception as exc:
        fail("api ping", f"{type(exc).__name__}: {exc}")


def check_registry() -> None:
    p = pathlib.Path(config.CASE_REGISTRY)
    if not p.exists():
        warn("registry", f"{p} missing; needed for run_tests.py, not for the app")
        return
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        warn("registry", "empty template; populate before a batch")
        return
    counts = Counter((r.get("pathway"), r.get("reference_stratum")) for r in rows)
    short = []
    for pathway in config.PATHWAYS:
        for stratum in config.STRATA_BY_PATHWAY[pathway]:
            n = counts.get((pathway, stratum), 0)
            if n != config.N_PER_STRATUM:
                short.append(f"{pathway}/{stratum}={n}")
    bad = [r["case_id"] for r in rows
           if r.get("pathway") == "Nevus"
           and r.get("reference_stratum") not in config.NEVUS_STRATA]
    if bad:
        fail("registry", f"{len(bad)} Nevus row(s) whose reference_stratum is "
             f"not an MPATH-Dx class (legacy mild/moderate/severe?): "
             f"{', '.join(bad[:3])}")
    missing_tiles = sum(1 for r in rows if not all(
        r.get(f"tile_{m}") for m in config.MAGNIFICATIONS))
    if missing_tiles:
        warn("registry", f"{missing_tiles} case(s) missing one or more tiles")
    if short:
        warn("registry", f"{len(rows)} rows; strata off target: "
             + ", ".join(short[:6]))
    else:
        ok("registry", f"{len(rows)} rows, every stratum at "
           f"{config.N_PER_STRATUM}")


def check_ground_truth() -> None:
    required = {"CSCC": {"case_id", "reference_grade"},
                "Nevus": {"case_id", "reference_class"}}
    for pathway in config.PATHWAYS:
        p = pathlib.Path(config.GROUND_TRUTH[pathway])
        if not p.exists():
            warn(f"ground truth {pathway}", f"{p} missing; needed for scoring")
            continue
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            rows = list(reader)
        missing = required[pathway] - cols
        if missing:
            fail(f"ground truth {pathway}", f"missing columns {sorted(missing)}")
        elif any("example row" in (r.get("notes") or "") for r in rows):
            warn(f"ground truth {pathway}", "still contains template example "
                 "rows; delete them before scoring")
        else:
            ok(f"ground truth {pathway}", f"{len(rows)} rows")


def check_log_dir() -> None:
    root = pathlib.Path(config.LOG_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_test"
        probe.write_text("x")
        probe.unlink()
        ok("log dir", f"{root}/ writable")
    except OSError as exc:
        fail("log dir", f"{root}/ not writable: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ping", action="store_true",
                    help="make one cheap API call to confirm model access")
    args = ap.parse_args()

    check_python()
    check_imports()
    check_sqlite()
    manifest = check_manifest()
    check_stores(manifest)
    check_key(args.ping)
    check_registry()
    check_ground_truth()
    check_log_dir()

    width = max(len(n) for _, n, _ in _results)
    for level, name, detail in _results:
        mark = {"ok": " ok ", "warn": "warn", "FAIL": "FAIL"}[level]
        print(f"[{mark}] {name:{width}}  {detail}")
    fails = sum(1 for l, _, _ in _results if l == "FAIL")
    warns = sum(1 for l, _, _ in _results if l == "warn")
    print(f"\n{fails} failure(s), {warns} warning(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
