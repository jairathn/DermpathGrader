"""
verify_logging.py
-----------------
Audits run_manifest.json and any supplied case log files against the spec.

    python verify_logging.py                              # manifest only
    python verify_logging.py --log path/to/rep1.json ...  # + case logs

Exit 0 = all checks PASS (WARNs are informational, not failures).
Exit 1 = one or more FAIL.

Checks are numbered so failures cite a specific ID.
"""

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── check accumulator ─────────────────────────────────────────────────────────

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"

_checks: List[Tuple[str, str, str]] = []   # (id, status, message)
_fail_count = 0
_warn_count = 0


def check(cid: str, condition: bool, msg_pass: str, msg_fail: str,
          level: str = FAIL):
    global _fail_count, _warn_count
    if condition:
        _checks.append((cid, PASS, msg_pass))
    else:
        _checks.append((cid, level, msg_fail))
        if level == FAIL:
            _fail_count += 1
        elif level == WARN:
            _warn_count += 1


def warn(cid: str, condition: bool, msg_pass: str, msg_warn: str):
    """Helper: always WARN (never FAIL) if condition is False."""
    check(cid, condition, msg_pass, msg_warn, level=WARN)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(s: str) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(
        c in "0123456789abcdef" for c in s.lower())


def print_results(verbose: bool = False):
    for cid, status, msg in _checks:
        if status != PASS or verbose:
            sym = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}.get(status, "?")
            print(f"  {sym}  [{cid}] {msg}")
    total = len(_checks)
    passed = sum(1 for _, s, _ in _checks if s == PASS)
    print(f"\n{'─'*70}")
    print(f"  {total} checks   {passed} PASS   {_fail_count} FAIL   {_warn_count} WARN")


# ── manifest checks ───────────────────────────────────────────────────────────

EXPECTED = {
    "CSCC":  {"chunk_count": 72,  "result_count": 35, "max_tokens": 1500,
              "schema": ["primary_grade","confidence_level","keratinization_present",
                         "atypia_level","key_features","additional_observations"]},
    "Nevus": {"chunk_count": 294, "result_count": 60, "max_tokens": 1500,
              "schema": ["traditional_grade","mpath_grade","confidence_level",
                         "nuclear_abnormality_count","architectural_features",
                         "cytological_features","grading_rationale",
                         "clinical_significance"]},
}

SHA256_EMPTY = _sha256("")   # e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855


def check_manifest(m: dict) -> dict:
    """Run all manifest checks. Returns {pathway: context_block_sha256}."""

    # ── top-level structure ───────────────────────────────────────────────────
    check("M01", m.get("manifest_version") == "1.0",
          "manifest_version == 1.0",
          f"manifest_version expected '1.0', got {m.get('manifest_version')!r}")

    check("M02", bool(m.get("session_id")),
          "session_id present and non-empty",
          "session_id missing or empty")

    check("M03", bool(m.get("created_utc")),
          "created_utc present",
          "created_utc missing")

    # environment
    env = m.get("environment", {})
    check("M04", bool(env.get("python_version")),
          "environment.python_version present",
          "environment.python_version missing")
    check("M05", bool(env.get("platform")),
          "environment.platform present",
          "environment.platform missing")

    pkgs = env.get("packages", {})
    required_pkgs = ["chromadb","sentence-transformers","langchain",
                     "langchain-text-splitters","pypdf","anthropic","pillow"]
    for i, pkg in enumerate(required_pkgs, 6):
        check(f"M0{i}", pkg in pkgs,
              f"packages.{pkg} present",
              f"packages.{pkg} missing from environment.packages")

    check("M13", bool(m.get("pathways", {}).get("CSCC")),
          "pathways.CSCC present",
          "pathways.CSCC missing")
    check("M14", bool(m.get("pathways", {}).get("Nevus")),
          "pathways.Nevus present",
          "pathways.Nevus missing")

    # ── per-pathway checks ────────────────────────────────────────────────────
    context_shas = {}
    for pw, exp in EXPECTED.items():
        p = m.get("pathways", {}).get(pw, {})
        pfx = f"{pw[0]}"   # C or N for prefix

        vs = p.get("vector_store", {})
        check(f"{pfx}V01", bool(vs.get("path")),
              f"{pw} vector_store.path present",
              f"{pw} vector_store.path missing")
        check(f"{pfx}V02", bool(vs.get("collection")),
              f"{pw} vector_store.collection present",
              f"{pw} vector_store.collection missing")
        check(f"{pfx}V03", vs.get("chunk_count") == exp["chunk_count"],
              f"{pw} chunk_count == {exp['chunk_count']}",
              f"{pw} chunk_count expected {exp['chunk_count']}, got {vs.get('chunk_count')}")
        check(f"{pfx}V04", _is_sha256(vs.get("chunk_id_fingerprint_sha256","")),
              f"{pw} chunk_id_fingerprint_sha256 is 64-char hex",
              f"{pw} chunk_id_fingerprint_sha256 missing or malformed")
        check(f"{pfx}V05", _is_sha256(vs.get("corpus_fingerprint_sha256","")),
              f"{pw} corpus_fingerprint_sha256 is 64-char hex",
              f"{pw} corpus_fingerprint_sha256 missing or malformed")

        ef = vs.get("embedding_function", {})
        check(f"{pfx}V06", bool(ef.get("class")),
              f"{pw} embedding_function.class present",
              f"{pw} embedding_function.class missing or empty")
        check(f"{pfx}V07", bool(ef.get("resolved_model")),
              f"{pw} embedding_function.resolved_model present",
              f"{pw} embedding_function.resolved_model missing — must be read at runtime, not hardcoded")
        check(f"{pfx}V08", isinstance(ef.get("dimensions"), int) and ef.get("dimensions", 0) > 0,
              f"{pw} embedding_function.dimensions = {ef.get('dimensions')}",
              f"{pw} embedding_function.dimensions missing or zero")

        chunking = vs.get("chunking", {})
        check(f"{pfx}V09", bool(chunking.get("splitter")),
              f"{pw} chunking.splitter present",
              f"{pw} chunking.splitter missing")
        check(f"{pfx}V10", isinstance(chunking.get("chunk_size"), int) and chunking["chunk_size"] > 0,
              f"{pw} chunking.chunk_size = {chunking.get('chunk_size')}",
              f"{pw} chunking.chunk_size missing or zero")

        src_docs = vs.get("source_documents", [])
        check(f"{pfx}V11", len(src_docs) > 0,
              f"{pw} source_documents has {len(src_docs)} entries",
              f"{pw} source_documents is empty")
        for di, doc in enumerate(src_docs):
            check(f"{pfx}V12.{di}", bool(doc.get("filename")),
                  f"{pw} source_documents[{di}].filename present",
                  f"{pw} source_documents[{di}].filename empty")
            check(f"{pfx}V13.{di}", _is_sha256(doc.get("sha256","")),
                  f"{pw} source_documents[{di}].sha256 is 64-char hex",
                  f"{pw} source_documents[{di}].sha256 missing or malformed")

        # retrieval
        ret = p.get("retrieval", {})
        check(f"{pfx}R01", ret.get("n_results_per_subquery") == 5,
              f"{pw} n_results_per_subquery == 5",
              f"{pw} n_results_per_subquery = {ret.get('n_results_per_subquery')} (expected 5)")

        subqs = ret.get("subqueries", [])
        check(f"{pfx}R02", len(subqs) > 0,
              f"{pw} subqueries list has {len(subqs)} entries",
              f"{pw} subqueries list is empty")

        results = ret.get("results", [])
        check(f"{pfx}R03", len(results) == exp["result_count"],
              f"{pw} retrieval.results has {len(results)} entries (expected {exp['result_count']})",
              f"{pw} retrieval.results has {len(results)} entries, expected {exp['result_count']}")

        required_result_keys = {"subquery_n","subquery","rank","chunk_id",
                                 "distance","source","page","text"}
        missing_keys_any = False
        empty_chunk_ids  = 0
        empty_texts      = 0
        bad_distances    = 0
        for ri, r in enumerate(results):
            missing = required_result_keys - set(r.keys())
            if missing:
                missing_keys_any = True
            if not r.get("chunk_id"):
                empty_chunk_ids += 1
            if not r.get("text"):
                empty_texts += 1
            d = r.get("distance")
            if not isinstance(d, (int, float)) or d != d:  # NaN check
                bad_distances += 1

        check(f"{pfx}R04", not missing_keys_any,
              f"{pw} all result entries have required keys",
              f"{pw} some result entries are missing keys (subquery_n/rank/chunk_id/distance/source/page/text)")
        check(f"{pfx}R05", empty_chunk_ids == 0,
              f"{pw} all result chunk_ids non-empty",
              f"{pw} {empty_chunk_ids} result(s) have empty chunk_id — cannot verify case-level chunk list")
        check(f"{pfx}R06", empty_texts == 0,
              f"{pw} all result texts non-empty",
              f"{pw} {empty_texts} result(s) have empty text field")
        check(f"{pfx}R07", bad_distances == 0,
              f"{pw} all distances are finite numbers",
              f"{pw} {bad_distances} distance(s) are non-numeric or NaN")

        unique_ids = ret.get("unique_chunk_ids", [])
        unique_count = ret.get("unique_chunk_count", -1)
        dup_count    = ret.get("duplicate_retrieval_count", -1)
        check(f"{pfx}R08", len(unique_ids) > 0,
              f"{pw} unique_chunk_ids non-empty ({len(unique_ids)} unique)",
              f"{pw} unique_chunk_ids is empty")
        check(f"{pfx}R09", unique_count == len(unique_ids),
              f"{pw} unique_chunk_count == len(unique_chunk_ids) == {unique_count}",
              f"{pw} unique_chunk_count {unique_count} != len(unique_chunk_ids) {len(unique_ids)}")
        check(f"{pfx}R10", dup_count >= 0,
              f"{pw} duplicate_retrieval_count = {dup_count}",
              f"{pw} duplicate_retrieval_count missing or negative")
        check(f"{pfx}R11", ret.get("context_block_chars", 0) > 0,
              f"{pw} context_block_chars = {ret.get('context_block_chars')}",
              f"{pw} context_block_chars is zero or missing")

        stored_sha = ret.get("context_block_sha256", "")
        check(f"{pfx}R12", _is_sha256(stored_sha),
              f"{pw} context_block_sha256 is 64-char hex",
              f"{pw} context_block_sha256 missing or malformed — this is the case-invariance anchor")

        # ── HIGH VALUE: reproducibility of context_block_sha256 ──────────────
        rebuilt_context = "\n\n".join(r.get("text","") for r in results)
        rebuilt_sha     = _sha256(rebuilt_context)
        check(f"{pfx}R13", rebuilt_sha == stored_sha,
              f"{pw} context_block_sha256 reproducible from results texts",
              f"{pw} context_block_sha256 NOT reproducible — "
              f"stored={stored_sha[:16]}… rebuilt={rebuilt_sha[:16]}… "
              "The case-invariance claim cannot be verified without this.")

        context_shas[pw] = stored_sha

        # generation
        gen = p.get("generation", {})
        check(f"{pfx}G01", bool(gen.get("model_requested")),
              f"{pw} generation.model_requested = {gen.get('model_requested')!r}",
              f"{pw} generation.model_requested missing")
        check(f"{pfx}G02", gen.get("temperature") == 0.1,
              f"{pw} generation.temperature == 0.1",
              f"{pw} generation.temperature = {gen.get('temperature')} (expected 0.1)")
        check(f"{pfx}G03", isinstance(gen.get("max_tokens"), int) and gen["max_tokens"] >= 1500,
              f"{pw} generation.max_tokens = {gen.get('max_tokens')} (>= 1500)",
              f"{pw} generation.max_tokens = {gen.get('max_tokens')} — "
              "values < 1500 risk truncation on complex schemas, raising the fallback rate")

        sys_tpl  = gen.get("system_template_text",  "")
        user_tpl = gen.get("user_template_text",    "")
        sys_sha  = gen.get("system_template_sha256", "")
        user_sha = gen.get("user_template_sha256",   "")

        check(f"{pfx}G04", isinstance(sys_tpl,  str),
              f"{pw} system_template_text is a string (may be empty)",
              f"{pw} system_template_text is not a string")
        check(f"{pfx}G05", bool(user_tpl),
              f"{pw} user_template_text non-empty",
              f"{pw} user_template_text is empty — prompt template was not extracted")

        # ── HIGH VALUE: system and user templates must NOT be identical ───────
        check(f"{pfx}G06", sys_tpl != user_tpl,
              f"{pw} system_template_text != user_template_text (not flattened into one string)",
              f"{pw} system_template_text == user_template_text — "
              "both are identical, which means the split was not preserved; "
              "SHA-256 hashes cannot be compared against the API boundary")

        check(f"{pfx}G07", _sha256(sys_tpl) == sys_sha,
              f"{pw} system_template_sha256 reproduces from system_template_text",
              f"{pw} system_template_sha256 does not match sha256(system_template_text)")
        check(f"{pfx}G08", _sha256(user_tpl) == user_sha,
              f"{pw} user_template_sha256 reproduces from user_template_text",
              f"{pw} user_template_sha256 does not match sha256(user_template_text)")

        # output_schema_fields lives at the pathway level, not inside generation
        schema = p.get("output_schema_fields", [])
        check(f"{pfx}G09", set(schema) == set(exp["schema"]),
              f"{pw} output_schema_fields has all {len(exp['schema'])} required fields",
              f"{pw} output_schema_fields mismatch  "
              f"expected={sorted(exp['schema'])}  got={sorted(schema)}")

    return context_shas


# ── per-case log checks ───────────────────────────────────────────────────────

CSCC_PARSED_SCHEMA = {
    "primary_grade": str, "confidence_level": str,
    "keratinization_present": (bool, type(None)),
    "atypia_level": (str, type(None)),
    "key_features": list, "additional_observations": (str, type(None)),
}

NEVI_PARSED_SCHEMA = {
    "traditional_grade": str, "mpath_grade": str, "confidence_level": str,
    "nuclear_abnormality_count": (int, float),
    "architectural_features": list, "cytological_features": list,
    "grading_rationale": (str, type(None)),
    "clinical_significance": (str, type(None)),
}

VALID_PARSE_STRATEGIES = {
    "direct_json", "regex", "keyword_inference", "keyword_fallback",
}


def check_log(log: dict, log_path: str, manifest_session_id: str,
              context_shas: Dict[str, str]):
    pfx = log_path.replace("/", "_").replace("\\", "_")[:30]

    # ── envelope ─────────────────────────────────────────────────────────────
    check(f"[{pfx}] L01", log.get("log_version") == "1.0",
          "log_version == 1.0",
          f"log_version expected '1.0', got {log.get('log_version')!r}")

    check(f"[{pfx}] L02", log.get("manifest_session_id") == manifest_session_id,
          f"manifest_session_id matches manifest ({manifest_session_id[:20]}…)",
          f"manifest_session_id mismatch: log={log.get('manifest_session_id')!r} "
          f"manifest={manifest_session_id!r}")

    pathway = log.get("pathway", "")
    check(f"[{pfx}] L03", pathway in ("CSCC", "Nevus"),
          f"pathway = {pathway!r}",
          f"pathway {pathway!r} not in [CSCC, Nevus]")

    check(f"[{pfx}] L04", bool(log.get("case_id")),
          f"case_id = {log.get('case_id')!r}",
          "case_id missing or empty")

    check(f"[{pfx}] L05", isinstance(log.get("replicate"), int) and log["replicate"] >= 1,
          f"replicate = {log.get('replicate')}",
          f"replicate must be int >= 1, got {log.get('replicate')!r}")

    check(f"[{pfx}] L06", bool(log.get("timestamp_utc")),
          "timestamp_utc present",
          "timestamp_utc missing")

    # ── image ─────────────────────────────────────────────────────────────────
    img = log.get("image", {})
    check(f"[{pfx}] L07", "source_registry" in img,
          "image.source_registry key present",
          "image.source_registry key missing")
    check(f"[{pfx}] L08", bool(img.get("source_filename")),
          f"image.source_filename = {img.get('source_filename')!r}",
          "image.source_filename missing or empty")
    check(f"[{pfx}] L09", _is_sha256(img.get("source_sha256", "")),
          "image.source_sha256 is 64-char hex",
          "image.source_sha256 missing or malformed")

    sdims = img.get("source_dimensions_px", [0, 0])
    check(f"[{pfx}] L10",
          isinstance(sdims, list) and len(sdims) == 2 and all(d > 0 for d in sdims),
          f"image.source_dimensions_px = {sdims}",
          f"image.source_dimensions_px invalid: {sdims}")

    check(f"[{pfx}] L11", img.get("sent_media_type") in ("image/jpeg", "image/png"),
          f"image.sent_media_type = {img.get('sent_media_type')!r}",
          f"image.sent_media_type {img.get('sent_media_type')!r} not in [image/jpeg, image/png]")

    tdims = img.get("sent_dimensions_px", [0, 0])
    check(f"[{pfx}] L12",
          isinstance(tdims, list) and len(tdims) == 2 and all(d > 0 for d in tdims),
          f"image.sent_dimensions_px = {tdims}",
          f"image.sent_dimensions_px invalid: {tdims}")

    check(f"[{pfx}] L13", isinstance(img.get("sent_bytes"), int) and img["sent_bytes"] > 0,
          f"image.sent_bytes = {img.get('sent_bytes')}",
          f"image.sent_bytes must be int > 0, got {img.get('sent_bytes')!r}")

    check(f"[{pfx}] L14", _is_sha256(img.get("sent_sha256", "")),
          "image.sent_sha256 is 64-char hex",
          "image.sent_sha256 missing or malformed — "
          "without this, process_image_file vs Streamlit path cannot be compared")

    check(f"[{pfx}] L15", isinstance(img.get("resize_applied"), bool),
          f"image.resize_applied is bool ({img.get('resize_applied')})",
          "image.resize_applied is not a bool")

    # ── retrieval ─────────────────────────────────────────────────────────────
    ret = log.get("retrieval", {})
    subqs = ret.get("subqueries_issued", [])
    check(f"[{pfx}] L16", isinstance(subqs, list) and len(subqs) > 0,
          f"retrieval.subqueries_issued has {len(subqs)} entries",
          "retrieval.subqueries_issued is empty or missing")

    chunk_ids = ret.get("retrieved_chunk_ids_in_order", [])
    check(f"[{pfx}] L17", isinstance(chunk_ids, list) and len(chunk_ids) > 0,
          f"retrieval.retrieved_chunk_ids_in_order has {len(chunk_ids)} entries",
          "retrieval.retrieved_chunk_ids_in_order is empty or missing")

    log_ctx_sha = ret.get("context_block_sha256", "")
    check(f"[{pfx}] L18", _is_sha256(log_ctx_sha),
          f"retrieval.context_block_sha256 is 64-char hex",
          "retrieval.context_block_sha256 missing or malformed — "
          "this is the per-case invariance anchor")

    # ── HIGH VALUE: case context hash must match manifest ─────────────────────
    manifest_ctx_sha = context_shas.get(pathway, "")
    check(f"[{pfx}] L19",
          bool(manifest_ctx_sha) and log_ctx_sha == manifest_ctx_sha,
          f"retrieval.context_block_sha256 matches manifest for {pathway}",
          f"retrieval.context_block_sha256 MISMATCH — "
          f"case={log_ctx_sha[:16]}… manifest={manifest_ctx_sha[:16]}… "
          "Case-invariant retrieval claim is broken for this case.")

    check(f"[{pfx}] L20", ret.get("matches_manifest_context") is True,
          "retrieval.matches_manifest_context == True",
          f"retrieval.matches_manifest_context = {ret.get('matches_manifest_context')!r}")

    # ── request ───────────────────────────────────────────────────────────────
    req = log.get("request", {})
    check(f"[{pfx}] L21", bool(req.get("model")),
          f"request.model = {req.get('model')!r}",
          "request.model missing or empty")

    check(f"[{pfx}] L22", req.get("temperature") == 0.1,
          "request.temperature == 0.1",
          f"request.temperature = {req.get('temperature')!r}")

    check(f"[{pfx}] L23", isinstance(req.get("max_tokens"), int) and req["max_tokens"] > 0,
          f"request.max_tokens = {req.get('max_tokens')}",
          f"request.max_tokens invalid: {req.get('max_tokens')!r}")

    sys_text  = req.get("system_text",  "")
    user_text = req.get("user_text",    "")
    sys_sha   = req.get("system_sha256", "")
    user_sha  = req.get("user_sha256",   "")

    # ── HIGH VALUE: system and user prompt split preserved ───────────────────
    check(f"[{pfx}] L24", sys_text != user_text,
          "request.system_text != request.user_text (split preserved)",
          "request.system_text == request.user_text — "
          "both fields are identical, indicating the API boundary was not preserved; "
          "SHA-256s cannot be compared against the actual API call")

    # system_sha256 reproducibility (even when system_text is empty)
    check(f"[{pfx}] L25", _sha256(sys_text) == sys_sha,
          f"request.system_sha256 reproduces from system_text",
          f"request.system_sha256 does not match sha256(system_text)  "
          f"stored={sys_sha[:16]}… expected={_sha256(sys_text)[:16]}…")

    check(f"[{pfx}] L26", bool(user_text),
          "request.user_text non-empty",
          "request.user_text is empty — prompt was not logged")

    # ── HIGH VALUE: user_sha256 reproducibility ───────────────────────────────
    check(f"[{pfx}] L27", _sha256(user_text) == user_sha,
          "request.user_sha256 reproduces from user_text",
          f"request.user_sha256 does not match sha256(user_text)  "
          f"stored={user_sha[:16]}… expected={_sha256(user_text)[:16]}…")

    # message_structure
    ms = req.get("message_structure", [])
    check(f"[{pfx}] L28", isinstance(ms, list) and len(ms) == 1,
          f"request.message_structure is list with 1 entry",
          f"request.message_structure has {len(ms) if isinstance(ms,list) else 'non-list'} entries (expected 1)")

    if isinstance(ms, list) and len(ms) == 1:
        msg = ms[0]
        check(f"[{pfx}] L29", msg.get("role") == "user",
              "message_structure[0].role == 'user'",
              f"message_structure[0].role = {msg.get('role')!r}")

        blocks = msg.get("blocks", [])
        block_types = {b.get("type") for b in blocks}
        check(f"[{pfx}] L30", "text" in block_types and "image" in block_types,
              f"message_structure blocks contain text and image (found {sorted(block_types)})",
              f"message_structure blocks missing text or image: {sorted(block_types)}")

        text_blocks  = [b for b in blocks if b.get("type") == "text"]
        image_blocks = [b for b in blocks if b.get("type") == "image"]

        if text_blocks:
            tb_sha = text_blocks[0].get("sha256", "")
            check(f"[{pfx}] L31", tb_sha == user_sha,
                  "message_structure text block sha256 == request.user_sha256",
                  f"message_structure text block sha256 {tb_sha[:16]}… "
                  f"!= user_sha256 {user_sha[:16]}… — block hash not reproducible")

        if image_blocks:
            ib_sha = image_blocks[0].get("sha256", "")
            sent_sha = img.get("sent_sha256", "")
            check(f"[{pfx}] L32", _is_sha256(ib_sha),
                  f"message_structure image block sha256 is 64-char hex",
                  "message_structure image block sha256 missing or malformed")
            check(f"[{pfx}] L33", ib_sha == sent_sha,
                  "message_structure image sha256 == image.sent_sha256",
                  f"image block sha256 {ib_sha[:16]}… != sent_sha256 {sent_sha[:16]}… "
                  "— the image logged in the envelope doesn't match the one sent")

    check(f"[{pfx}] L34", isinstance(req.get("attempt"), int) and req["attempt"] >= 1,
          f"request.attempt = {req.get('attempt')}",
          f"request.attempt must be int >= 1, got {req.get('attempt')!r}")

    check(f"[{pfx}] L35", isinstance(req.get("prior_attempt_errors"), list),
          "request.prior_attempt_errors is list",
          "request.prior_attempt_errors is not a list")

    # ── response ──────────────────────────────────────────────────────────────
    resp = log.get("response", {})

    # ── HIGH VALUE: api_request_id, model_returned, stop_reason ──────────────
    check(f"[{pfx}] L36", bool(resp.get("api_request_id")),
          f"response.api_request_id = {str(resp.get('api_request_id'))[:20]!r}",
          "response.api_request_id missing or empty — "
          "cannot cross-reference with Anthropic's request logs")

    check(f"[{pfx}] L37", bool(resp.get("model_returned")),
          f"response.model_returned = {resp.get('model_returned')!r}",
          "response.model_returned missing — "
          "must be read from the response body, not copied from the request")

    check(f"[{pfx}] L38", bool(resp.get("stop_reason")),
          f"response.stop_reason = {resp.get('stop_reason')!r}",
          "response.stop_reason missing or empty")

    warn(f"[{pfx}] L39", resp.get("stop_reason") != "max_tokens",
         f"response.stop_reason != max_tokens (= {resp.get('stop_reason')!r})",
         f"response.stop_reason == 'max_tokens' — output was truncated; "
         "JSON is likely malformed and parser fallback fired. "
         "Raise max_tokens before running remaining cases.")

    usage = resp.get("usage", {})
    check(f"[{pfx}] L40", isinstance(usage.get("input_tokens"), int) and usage["input_tokens"] > 0,
          f"response.usage.input_tokens = {usage.get('input_tokens')}",
          f"response.usage.input_tokens invalid: {usage.get('input_tokens')!r}")

    check(f"[{pfx}] L41", isinstance(usage.get("output_tokens"), int) and usage["output_tokens"] > 0,
          f"response.usage.output_tokens = {usage.get('output_tokens')}",
          f"response.usage.output_tokens invalid: {usage.get('output_tokens')!r}")

    max_tok = req.get("max_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    warn(f"[{pfx}] L42",
         not (isinstance(out_tok, int) and isinstance(max_tok, int)
              and max_tok > 0 and out_tok / max_tok > 0.90),
         f"output_tokens {out_tok} < 90% of max_tokens {max_tok}",
         f"output_tokens {out_tok} is {round(100*out_tok/max_tok if max_tok else 0)}% of "
         f"max_tokens {max_tok} — dangerously close to ceiling; raise max_tokens now")

    check(f"[{pfx}] L43", isinstance(resp.get("latency_ms"), int) and resp["latency_ms"] > 0,
          f"response.latency_ms = {resp.get('latency_ms')} ms",
          f"response.latency_ms invalid: {resp.get('latency_ms')!r}")

    check(f"[{pfx}] L44", bool(resp.get("raw_text")),
          "response.raw_text non-empty (verbatim model output logged)",
          "response.raw_text is empty — model output was not logged byte-for-byte")

    # ── parsing ───────────────────────────────────────────────────────────────
    pars = log.get("parsing", {})
    strategy = pars.get("strategy_used", "")
    check(f"[{pfx}] L45", strategy in VALID_PARSE_STRATEGIES,
          f"parsing.strategy_used = {strategy!r}",
          f"parsing.strategy_used {strategy!r} not in {sorted(VALID_PARSE_STRATEGIES)}")

    warn(f"[{pfx}] L46", strategy != "keyword_inference",
         f"parsing.strategy_used = {strategy!r} (structured JSON)",
         f"parsing.strategy_used = 'keyword_inference' — "
         "grade was inferred from prose keywords, not the model's structured output; "
         "this case must be flagged in parser_fallback_summary.csv")

    check(f"[{pfx}] L47", isinstance(pars.get("fallback_invoked"), bool),
          f"parsing.fallback_invoked = {pars.get('fallback_invoked')}",
          "parsing.fallback_invoked is not a bool")

    check(f"[{pfx}] L48", isinstance(pars.get("parse_errors"), list),
          "parsing.parse_errors is a list",
          "parsing.parse_errors is not a list")

    parsed = pars.get("parsed", {})
    schema = CSCC_PARSED_SCHEMA if pathway == "CSCC" else NEVI_PARSED_SCHEMA
    missing_fields = [k for k in schema if k not in parsed]
    check(f"[{pfx}] L49", len(missing_fields) == 0,
          f"parsing.parsed has all {len(schema)} schema fields",
          f"parsing.parsed missing fields: {missing_fields}")

    for field, expected_type in schema.items():
        if field in parsed:
            val = parsed[field]
            ok = isinstance(val, expected_type) if isinstance(expected_type, type) \
                 else isinstance(val, expected_type)
            check(f"[{pfx}] L50.{field}", ok,
                  f"parsed.{field} has correct type ({type(val).__name__})",
                  f"parsed.{field} = {val!r} has wrong type "
                  f"(expected {expected_type}, got {type(val).__name__})")

    # errors list
    check(f"[{pfx}] L51", isinstance(log.get("errors"), list),
          "errors is a list",
          "errors field is not a list")

    warn(f"[{pfx}] L52", len(log.get("errors", [])) == 0,
         "errors list is empty",
         f"errors list has {len(log.get('errors',[]))} entries: {log.get('errors',[])} — "
         "review before including this case")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify logging spec compliance")
    parser.add_argument("--log",     action="append", default=[],
                        metavar="PATH",
                        help="Path to a case log JSON (repeat for multiple)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print PASS lines too (default: FAIL+WARN only)")
    args = parser.parse_args()

    print("\n🔬  verify_logging.py")
    print("─" * 70)

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest_path = pathlib.Path("run_manifest.json")
    print(f"\n  Checking manifest: {manifest_path}")

    check("M00", manifest_path.exists(),
          "run_manifest.json exists",
          "run_manifest.json not found — run make_manifest.py first")

    if not manifest_path.exists():
        print_results(args.verbose)
        sys.exit(1)

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        check("M00b", False, "", f"run_manifest.json is not valid JSON: {e}")
        print_results(args.verbose)
        sys.exit(1)

    check("M00b", True, "run_manifest.json is valid JSON", "")
    session_id    = manifest.get("session_id", "")
    context_shas  = check_manifest(manifest)

    # ── case logs ─────────────────────────────────────────────────────────────
    log_paths = args.log
    if not log_paths:
        # Auto-discover all logs if none specified
        discovered = sorted(pathlib.Path("analysis_logs").rglob("*.json")) \
                     if pathlib.Path("analysis_logs").exists() else []
        if discovered:
            print(f"\n  No --log flags; auto-discovered {len(discovered)} log file(s)")
            log_paths = [str(p) for p in discovered]
        else:
            print("\n  No case logs supplied and none found in analysis_logs/.")
            print("  Run:  python run_tests.py  then re-run this verifier.")

    for log_path in log_paths:
        p = pathlib.Path(log_path)
        print(f"\n  Checking case log: {p}")
        check(f"[{p.name}] F01", p.exists(),
              f"{p} exists",
              f"{p} not found")
        if not p.exists():
            continue
        try:
            log = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            check(f"[{p.name}] F02", False, "",
                  f"{p} is not valid JSON: {e}")
            continue
        check(f"[{p.name}] F02", True, f"{p} is valid JSON", "")
        check_log(log, p.name, session_id, context_shas)

    # ── results ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print_results(args.verbose)

    if _fail_count > 0:
        print(f"\n  ❌  {_fail_count} check(s) FAILED.  Fix before running cases.")
        sys.exit(1)
    else:
        if _warn_count > 0:
            print(f"\n  ⚠️   {_warn_count} warning(s). Review before proceeding.")
        else:
            print(f"\n  ✅  All checks PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
