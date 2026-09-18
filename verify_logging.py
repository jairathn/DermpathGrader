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

import config
import pathlib
import sys
from typing import Dict, List, Tuple

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

# Field lists come from the analyzers so the verifier cannot drift away
# from the schema that actually ran.
from image_analyzer import OUTPUT_SCHEMA_FIELDS as CSCC_SCHEMA_FIELDS
from nevi_analyzer import OUTPUT_SCHEMA_FIELDS as NEVUS_SCHEMA_FIELDS

EXPECTED = {
    "CSCC":  {"chunk_count": 72,  "result_count": 35, "max_tokens": config.MAX_TOKENS,
              "schema": CSCC_SCHEMA_FIELDS},
    "Nevus": {"chunk_count": 294, "result_count": 60, "max_tokens": config.MAX_TOKENS,
              "schema": NEVUS_SCHEMA_FIELDS},
}

SHA256_EMPTY = _sha256("")   # e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855


def check_manifest(m: dict) -> dict:
    """Run all manifest checks. Returns {pathway: context_block_sha256}."""

    # ── top-level structure ───────────────────────────────────────────────────
    check("M01", m.get("manifest_version") == "2.0",
          "manifest_version == 2.0",
          f"manifest_version expected '2.0', got {m.get('manifest_version')!r}")
    check("M01b", m.get("protocol_version") == config.PROTOCOL_VERSION,
          f"manifest protocol_version == {config.PROTOCOL_VERSION}",
          f"manifest protocol_version = {m.get('protocol_version')!r}, code is "
          f"{config.PROTOCOL_VERSION!r}; re-run make_manifest.py --yes")

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
        repro = p.get("published_reproduction", {})
        if vs.get("chunk_count") == exp["chunk_count"]:
            check(f"{pfx}V03", True,
                  f"{pw} chunk_count == {exp['chunk_count']} (published value)", "")
        elif repro.get("accepted") and not repro.get("reproduced"):
            # Divergence was declared at manifest time and is on the record.
            warn(f"{pfx}V03", False,
                 "",
                 f"{pw} chunk_count is {vs.get('chunk_count')}, published value "
                 f"was {exp['chunk_count']}; divergence accepted and recorded in "
                 f"manifest.published_reproduction. Do not pool {pw} retrieval "
                 f"distances with pre-migration ones.")
        else:
            check(f"{pfx}V03", False, "",
                  f"{pw} chunk_count expected {exp['chunk_count']}, got "
                  f"{vs.get('chunk_count')}, and the divergence was not accepted "
                  f"at manifest time (make_manifest.py --accept-store-divergence)")
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
            if doc.get("substituted"):
                warn(f"{pfx}V14.{di}", False, "",
                     f"{pw} source_documents[{di}] ({doc.get('filename')}) is a "
                     f"TEXT SUBSTITUTE for a lost PDF; chunk boundaries differ "
                     f"from the original and every later chunk id has shifted")

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
        for _ri, r in enumerate(results):
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
        check(f"{pfx}G02", gen.get("temperature") is None,
              f"{pw} generation.temperature is null (not sent on this model family)",
              f"{pw} generation.temperature = {gen.get('temperature')!r}; sampling "
              "parameters are rejected by this model and must not be recorded "
              "as if applied")
        check(f"{pfx}G02b", gen.get("schema_enforced") is True
              and isinstance(gen.get("thinking"), dict),
              f"{pw} generation records schema_enforced and thinking",
              f"{pw} generation.schema_enforced={gen.get('schema_enforced')!r} "
              f"thinking={gen.get('thinking')!r}")
        check(f"{pfx}G03", isinstance(gen.get("max_tokens"), int)
              and gen["max_tokens"] >= config.MAX_TOKENS,
              f"{pw} generation.max_tokens = {gen.get('max_tokens')} "
              f"(>= {config.MAX_TOKENS})",
              f"{pw} generation.max_tokens = {gen.get('max_tokens')}; below "
              f"config.MAX_TOKENS={config.MAX_TOKENS} the v2.1 schema can truncate")

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
    "primary_grade": str, "broders_grade": int, "specimen_adequacy": str,
    "histologic_subtype": str, "depth_of_invasion": str,
    "high_risk_features": list, "confidence_level": str,
    "keratinization_present": (bool, type(None)),
    "atypia_level": (str, type(None)),
    "differential_diagnosis": list, "recommended_ancillary_studies": list,
    "key_features": list, "magnification_evidence": list,
    "additional_observations": (str, type(None)),
    "consistency_flags": list, "internally_consistent": bool,
}

NEVI_PARSED_SCHEMA = {
    "mpath_dx_v2_class": str, "lesion_category": str,
    "specimen_adequacy": str, "dysplasia_grade": str,
    "melanoma_subtype": str, "melanoma_histologic_subtype": str,
    "breslow_estimate_mm": (int, float, type(None)),
    "ulceration_present": (bool, type(None)),
    "mitoses_per_mm2": (int, float, type(None)),
    "confidence_level": str,
    "differential_diagnosis": list, "recommended_ancillary_studies": list,
    "architectural_features": list, "cytological_features": list,
    "magnification_evidence": list,
    "grading_rationale": (str, type(None)),
    "clinical_significance": (str, type(None)),
    "consistency_flags": list, "internally_consistent": bool,
}

VALID_PARSE_STRATEGIES = {
    "structured_output", "failed",
    # v1.0 strategies, kept so old logs still parse. Any of these appearing
    # in a v2.0 log means the schema was not enforced.
    "direct_json", "regex", "keyword_inference", "keyword_fallback",
}


def check_log(log: dict, log_path: str, manifest_session_id: str,
              context_shas: Dict[str, str]):
    pfx = log_path.replace("/", "_").replace("\\", "_")[:30]

    # ── envelope ─────────────────────────────────────────────────────────────
    log_version = log.get("log_version")
    check(f"[{pfx}] L01", log_version == config.LOG_VERSION,
          f"log_version == {config.LOG_VERSION}",
          f"log_version expected {config.LOG_VERSION!r}, got {log_version!r}. "
          "A v1.0 log holds one image and no magnification; it is not "
          "comparable to a v2.0 log and must not be pooled with one.")

    check(f"[{pfx}] L01b", log.get("protocol_version") == config.PROTOCOL_VERSION,
          f"protocol_version == {config.PROTOCOL_VERSION}",
          f"protocol_version = {log.get('protocol_version')!r}, expected "
          f"{config.PROTOCOL_VERSION!r}")

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

    # ── image set (protocol v2.0: four magnifications per case) ──────────────
    iset = log.get("image_set", {})
    images = log.get("images", [])

    check(f"[{pfx}] L07", "source_registry" in iset,
          "image_set.source_registry key present",
          "image_set.source_registry key missing")

    expected_mags = list(config.REQUIRED_MAGNIFICATIONS)
    sent_mags = iset.get("magnifications_sent", [])
    check(f"[{pfx}] L08", sent_mags == expected_mags,
          f"image_set.magnifications_sent == {expected_mags}",
          f"image_set.magnifications_sent = {sent_mags}, expected "
          f"{expected_mags}. A case graded on a different set of powers is "
          "not comparable to the rest of the batch.")

    check(f"[{pfx}] L09", len(images) == len(expected_mags),
          f"images has {len(expected_mags)} entries",
          f"images has {len(images)} entries, expected {len(expected_mags)}")

    check(f"[{pfx}] L10", _is_sha256(iset.get("set_sha256", "")),
          "image_set.set_sha256 is 64-char hex",
          "image_set.set_sha256 missing or malformed - without it two runs "
          "of the same case cannot be compared in one step")

    check(f"[{pfx}] L11",
          isinstance(iset.get("total_sent_bytes"), int)
          and iset["total_sent_bytes"] > 0,
          f"image_set.total_sent_bytes = {iset.get('total_sent_bytes')}",
          f"image_set.total_sent_bytes invalid: "
          f"{iset.get('total_sent_bytes')!r}")

    warn(f"[{pfx}] L12", bool(iset.get("tile_selection_method")),
         f"image_set.tile_selection_method = "
         f"{iset.get('tile_selection_method')!r}",
         "image_set.tile_selection_method is empty - the record cannot say "
         "who chose the 4x/10x/40x fields or whether they were blinded")

    # Per-image checks. These scale the check count with the magnification
    # set, which is the point: each frame the model saw must be accounted
    # for individually.
    for idx, img in enumerate(images):
        mag = img.get("magnification", f"#{idx}")
        ipfx = f"[{pfx}] L13.{idx}"

        check(f"{ipfx}a", mag in expected_mags,
              f"images[{idx}].magnification = {mag!r}",
              f"images[{idx}].magnification = {mag!r} not in {expected_mags}")

        check(f"{ipfx}b", bool(img.get("source_filename")),
              f"images[{idx}]({mag}).source_filename present",
              f"images[{idx}]({mag}).source_filename missing")

        check(f"{ipfx}c", _is_sha256(img.get("source_sha256", "")),
              f"images[{idx}]({mag}).source_sha256 is 64-char hex",
              f"images[{idx}]({mag}).source_sha256 missing or malformed")

        check(f"{ipfx}d", _is_sha256(img.get("sent_sha256", "")),
              f"images[{idx}]({mag}).sent_sha256 is 64-char hex",
              f"images[{idx}]({mag}).sent_sha256 missing or malformed - "
              "without it the UI and batch paths cannot be compared")

        sdims = img.get("source_dimensions_px", [0, 0])
        check(f"{ipfx}e",
              isinstance(sdims, list) and len(sdims) == 2
              and all(d > 0 for d in sdims),
              f"images[{idx}]({mag}).source_dimensions_px = {sdims}",
              f"images[{idx}]({mag}).source_dimensions_px invalid: {sdims}")

        tdims = img.get("sent_dimensions_px", [0, 0])
        check(f"{ipfx}f",
              isinstance(tdims, list) and len(tdims) == 2
              and all(d > 0 for d in tdims),
              f"images[{idx}]({mag}).sent_dimensions_px = {tdims}",
              f"images[{idx}]({mag}).sent_dimensions_px invalid: {tdims}")

        check(f"{ipfx}g",
              isinstance(tdims, list) and len(tdims) == 2
              and max(tdims) <= config.MAX_IMAGE_EDGE_PX,
              f"images[{idx}]({mag}) long edge <= "
              f"{config.MAX_IMAGE_EDGE_PX} px",
              f"images[{idx}]({mag}) long edge {max(tdims) if tdims else '?'} "
              f"exceeds {config.MAX_IMAGE_EDGE_PX} px - the API downsamples "
              "past this, so the extra pixels were paid for and discarded")

        check(f"{ipfx}h",
              img.get("sent_media_type") == config.OUTBOUND_MEDIA_TYPE,
              f"images[{idx}]({mag}).sent_media_type = "
              f"{img.get('sent_media_type')!r}",
              f"images[{idx}]({mag}).sent_media_type = "
              f"{img.get('sent_media_type')!r}, expected "
              f"{config.OUTBOUND_MEDIA_TYPE!r} - a mixed codec across cases "
              "makes them non-comparable")

        check(f"{ipfx}i",
              isinstance(img.get("sent_bytes"), int)
              and 0 < img["sent_bytes"] <= config.MAX_IMAGE_BYTES,
              f"images[{idx}]({mag}).sent_bytes = {img.get('sent_bytes')}",
              f"images[{idx}]({mag}).sent_bytes = {img.get('sent_bytes')!r} "
              f"outside (0, {config.MAX_IMAGE_BYTES}]")

        quality = img.get("compression_quality")
        check(f"{ipfx}j",
              isinstance(quality, int)
              and config.JPEG_QUALITY_FLOOR <= quality
              <= config.JPEG_QUALITY_START,
              f"images[{idx}]({mag}).compression_quality = {quality}",
              f"images[{idx}]({mag}).compression_quality = {quality!r}; must "
              f"be an int in [{config.JPEG_QUALITY_FLOOR}, "
              f"{config.JPEG_QUALITY_START}]. v1.0 logged None here, so the "
              "record could not say what the model actually saw.")

        check(f"{ipfx}k", isinstance(img.get("resize_applied"), bool),
              f"images[{idx}]({mag}).resize_applied is bool",
              f"images[{idx}]({mag}).resize_applied = "
              f"{img.get('resize_applied')!r}")

    # Consistency between the set summary and the per-image records.
    if images:
        check(f"[{pfx}] L14",
              iset.get("total_sent_bytes")
              == sum(i.get("sent_bytes", 0) for i in images),
              "image_set.total_sent_bytes == sum(images[].sent_bytes)",
              "image_set.total_sent_bytes disagrees with the per-image sum")

        check(f"[{pfx}] L14b",
              len({i.get("sent_sha256") for i in images}) == len(images),
              "all magnifications are distinct images",
              "two or more magnifications have the same sent_sha256 - the "
              "same frame was sent twice, so the case was not graded on "
              f"{len(expected_mags)} distinct powers")

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
          "retrieval.context_block_sha256 is 64-char hex",
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

    # This model family rejects sampling parameters, so v2.0 sends none.
    check(f"[{pfx}] L22", req.get("temperature") is None,
          "request.temperature is null (not sent on this model family)",
          f"request.temperature = {req.get('temperature')!r}; sampling "
          "parameters are rejected by this model and must not be logged as "
          "if they were applied")

    check(f"[{pfx}] L22b", isinstance(req.get("thinking"), dict)
          and req["thinking"].get("type") == "adaptive",
          "request.thinking is adaptive",
          f"request.thinking = {req.get('thinking')!r}, expected adaptive")

    check(f"[{pfx}] L22c", req.get("effort") == config.EFFORT,
          f"request.effort == {config.EFFORT!r}",
          f"request.effort = {req.get('effort')!r}, expected {config.EFFORT!r}")

    check(f"[{pfx}] L22d", req.get("schema_enforced") is True,
          "request.schema_enforced is True",
          "request.schema_enforced is not True - without server-side schema "
          "enforcement a malformed response can still reach the parser")

    check(f"[{pfx}] L22e", _is_sha256(req.get("output_schema_sha256", "")),
          "request.output_schema_sha256 is 64-char hex",
          "request.output_schema_sha256 missing - the record cannot say "
          "which schema version the grade was produced under")

    check(f"[{pfx}] L23", isinstance(req.get("max_tokens"), int) and req["max_tokens"] >= config.MAX_TOKENS,
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
          "request.system_sha256 reproduces from system_text",
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
    check(f"[{pfx}] L28", isinstance(ms, list) and len(ms) == 2
          and [m.get("role") for m in ms] == ["system", "user"],
          "request.message_structure is [system, user]",
          f"request.message_structure roles = "
          f"{[m.get('role') for m in ms] if isinstance(ms, list) else 'non-list'}"
          ", expected [system, user] (v2.1: cached scaffold in system)")

    if isinstance(ms, list) and len(ms) == 2:
        sys_blocks = ms[0].get("blocks", [])
        check(f"[{pfx}] L29", len(sys_blocks) == 1
              and sys_blocks[0].get("sha256") == sys_sha
              and sys_blocks[0].get("cached") is True,
              "system scaffold block sha256 == request.system_sha256, cached",
              "system scaffold block missing, uncached, or its sha256 does "
              "not match request.system_sha256")

        blocks = ms[1].get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        image_blocks = [b for b in blocks if b.get("type") == "image"]
        n_mags = len(config.REQUIRED_MAGNIFICATIONS)

        check(f"[{pfx}] L30", len(image_blocks) == n_mags,
              f"user turn has {n_mags} image blocks",
              f"user turn has {len(image_blocks)} image blocks, expected {n_mags}")

        check(f"[{pfx}] L30b", len(text_blocks) == n_mags + 1,
              f"user turn has {n_mags + 1} text blocks "
              f"({n_mags} captions + 1 instruction)",
              f"user turn has {len(text_blocks)} text blocks, expected "
              f"{n_mags + 1}. Unlabelled images in a multi-image request get "
              "conflated, so every image needs its caption.")

        instr = [b for b in text_blocks if b.get("role") == "instruction"]
        check(f"[{pfx}] L31", len(instr) == 1 and instr[0].get("sha256") == user_sha
              and blocks and blocks[-1] is instr[0],
              "instruction block is last and its sha256 == request.user_sha256",
              "instruction block missing, not last, or sha256 != user_sha256")

        logged_shas = [i.get("sent_sha256") for i in images]
        envelope_shas = [b.get("sha256") for b in image_blocks]
        check(f"[{pfx}] L32", all(_is_sha256(x or "") for x in envelope_shas),
              "all envelope image sha256 are 64-char hex",
              "one or more envelope image sha256 missing/malformed")
        check(f"[{pfx}] L33", envelope_shas == logged_shas,
              "envelope image sha256 list == images[].sent_sha256",
              f"envelope image hashes {[(x or '')[:8] for x in envelope_shas]} "
              f"!= logged {[(x or '')[:8] for x in logged_shas]} - the images "
              "in the envelope are not the images recorded for this case")
        envelope_mags = [b.get("magnification") for b in image_blocks]
        check(f"[{pfx}] L33b",
              envelope_mags == list(config.REQUIRED_MAGNIFICATIONS),
              f"envelope image order == {list(config.REQUIRED_MAGNIFICATIONS)}",
              f"envelope image order = {envelope_mags}; presentation order is "
              "part of the protocol and must not vary between cases")

    check(f"[{pfx}] L34", isinstance(req.get("attempt"), int) and req["attempt"] >= 1,
          f"request.attempt = {req.get('attempt')}",
          f"request.attempt must be int >= 1, got {req.get('attempt')!r}")

    check(f"[{pfx}] L35", isinstance(req.get("prior_attempt_errors"), list),
          "request.prior_attempt_errors is list",
          "request.prior_attempt_errors is not a list")

    # ── failed case: no response to check, but the record must be complete ──
    failure = log.get("failure")
    if failure:
        check(f"[{pfx}] F01", bool(failure.get("error_class")),
              f"failure.error_class = {failure.get('error_class')!r}",
              "failure.error_class missing - a failed case must say why")
        check(f"[{pfx}] F02", bool(failure.get("message")),
              "failure.message present",
              "failure.message missing")
        check(f"[{pfx}] F03",
              log.get("parsing", {}).get("strategy_used") == "failed"
              and not log.get("parsing", {}).get("parsed"),
              "parsing marks the case as failed with no parsed grade",
              "a failed case carries a parsed grade - the failure record and "
              "the parsing record disagree, and the scorer could pick it up")
        check(f"[{pfx}] F04", bool(req.get("user_sha256")) and bool(
            log.get("image_set", {}).get("set_sha256")),
              "failed case still records the request and image set attempted",
              "failed case is missing the request or image set - the attempt "
              "is not reproducible")
        if failure.get("error_class") == "GradingRefused":
            warn(f"[{pfx}] F05", False, "",
                 f"case was REFUSED by the model (category "
                 f"{failure.get('category')!r}); it has no grade and is "
                 "excluded from concordance. Report the refusal count.")
        else:
            warn(f"[{pfx}] F05", False, "",
                 f"case produced no grade ({failure.get('error_class')}); "
                 "excluded from concordance")
        return

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
         "response.stop_reason == 'max_tokens' — output was truncated; "
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
         "parsing.strategy_used = 'keyword_inference' — "
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
        root = pathlib.Path(config.LOG_ROOT)
        discovered = sorted(
            p for pw in config.PATHWAYS
            for p in (root / pw).glob("*__rep*.json")) if root.exists() else []
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
            print("\n  ✅  All checks PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
