"""Synoptic report from a case log.

    python report.py analysis_logs/Nevus/NEV-0001__rep1.json
    python report.py --all --outdir reports/

Renders one case log as a structured report in the shape a sign-out
would take: diagnosis line first, then the fields that drive
management, then the evidence, then provenance. It reads only the log,
never the ground truth, so a report can be handed to a reader without
un-blinding anything.

The footer carries what a reviewer would need to trust the number: the
model, the protocol, the attempt count, the schema hash, and the image
hashes. A report with no provenance is a claim; one with it is a record.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import config
import mpath_dx


def _line(label: str, value) -> str:
    if value is None or value == "" or value == []:
        return ""
    if isinstance(value, list):
        value = "; ".join(str(v).replace("_", " ") for v in value)
    elif isinstance(value, bool):
        value = "yes" if value else "no"
    else:
        value = str(value).replace("_", " ")
    return f"- **{label}:** {value}\n"


def _melanocytic(parsed: dict) -> str:
    cls = str(parsed.get("mpath_dx_v2_class") or "")
    label = mpath_dx.CLASS_DEFINITIONS.get(cls, {}).get("label", "")
    category = parsed.get("lesion_category", "")
    out = f"## Diagnosis\n\n**MPATH-Dx v2.0 Class {cls} - {label}**\n\n"

    if category == "melanoma":
        subtype = parsed.get("melanoma_subtype", "")
        text = "Melanoma"
        text += " in situ" if subtype == "in_situ" else (
            " (invasive)" if subtype == "invasive" else "")
        hist = parsed.get("melanoma_histologic_subtype")
        if hist and hist != "not_applicable":
            text += f", {hist.replace('_', ' ')} type"
        out += f"{text}\n\n"
        out += "### Staging features\n\n"
        out += _line("Breslow thickness (estimate)",
                     f"{parsed['breslow_estimate_mm']} mm"
                     if parsed.get("breslow_estimate_mm") is not None else None)
        out += _line("Ulceration", parsed.get("ulceration_present"))
        out += _line("Mitoses", f"{parsed['mitoses_per_mm2']} /mm2"
                     if parsed.get("mitoses_per_mm2") is not None else None)
    elif category == "dysplastic_nevus":
        out += (f"Dysplastic nevus, "
                f"{str(parsed.get('dysplasia_grade', '')).replace('_', ' ')} "
                f"atypia\n\n")
    elif category:
        out += f"{str(category).replace('_', ' ').capitalize()}\n\n"

    if cls and mpath_dx.is_valid_class(cls):
        mgmt = ("Re-excision indicated (Class II or above)."
                if mpath_dx.requires_reexcision(cls)
                else "No re-excision indicated on this basis (Class I).")
        out += f"### Management implication\n\n{mgmt}\n\n"
        if cls == "II":
            out += ("Class II covers both high-grade dysplastic nevi and "
                    "melanoma in situ; the diagnosis line above is what "
                    "distinguishes them.\n\n")

    out += "### Features\n\n"
    out += _line("Architectural", parsed.get("architectural_features"))
    out += _line("Cytological", parsed.get("cytological_features"))
    return out


def _cscc(parsed: dict) -> str:
    grade = parsed.get("primary_grade", "")
    out = f"## Diagnosis\n\n**Cutaneous squamous cell carcinoma, {grade.lower()}**\n\n"
    out += "### Grading and staging features\n\n"
    out += _line("Broders grade", parsed.get("broders_grade"))
    out += _line("Histologic subtype", parsed.get("histologic_subtype"))
    out += _line("Depth of invasion", parsed.get("depth_of_invasion"))
    out += _line("High-risk features", parsed.get("high_risk_features"))
    out += _line("Keratinization", parsed.get("keratinization_present"))
    out += _line("Atypia", parsed.get("atypia_level"))
    out += "\n### Features\n\n"
    out += _line("Key", parsed.get("key_features"))
    return out


def render(log: dict) -> str:
    """Markdown synoptic report for one case log."""
    pathway = log.get("pathway", "")
    parsed = log.get("parsing", {}).get("parsed", {}) or {}
    failure = log.get("failure")

    out = (f"# {pathway} case {log.get('case_id')} "
           f"(replicate {log.get('replicate')})\n\n")
    out += (f"Protocol v{log.get('protocol_version')} - "
            f"{log.get('request', {}).get('model')} - "
            f"{log.get('timestamp_utc')} UTC\n\n")

    if failure:
        out += "## No grade produced\n\n"
        out += _line("Reason", failure.get("error_class"))
        out += _line("Detail", failure.get("message"))
        if failure.get("category"):
            out += _line("Refusal category", failure["category"])
        return out + _provenance(log)

    out += _melanocytic(parsed) if pathway == "Nevus" else _cscc(parsed)

    out += "\n### Assessment quality\n\n"
    out += _line("Specimen adequacy", parsed.get("specimen_adequacy"))
    out += _line("Confidence", parsed.get("confidence_level"))
    flags = parsed.get("consistency_flags") or []
    if flags:
        out += ("- **Internal consistency:** FLAGGED - " +
                "; ".join(flags) + "\n")
    else:
        out += "- **Internal consistency:** consistent\n"

    out += "\n### Differential diagnosis\n\n"
    for i, d in enumerate(parsed.get("differential_diagnosis") or [], 1):
        out += f"{i}. {d}\n"
    out += "\n### Recommended ancillary studies\n\n"
    out += _line("Studies", parsed.get("recommended_ancillary_studies")) or "- none listed\n"

    out += "\n### Evidence by magnification\n\n"
    for item in parsed.get("magnification_evidence") or []:
        if isinstance(item, dict):
            out += f"- **{item.get('magnification')}:** {item.get('finding')}\n"

    rationale = parsed.get("grading_rationale") or parsed.get(
        "additional_observations")
    if rationale:
        out += f"\n### Rationale\n\n{rationale}\n"
    sig = parsed.get("clinical_significance")
    if sig:
        out += f"\n### Clinical significance\n\n{sig}\n"

    return out + _provenance(log)


def _provenance(log: dict) -> str:
    req = log.get("request", {})
    resp = log.get("response", {})
    iset = log.get("image_set", {})
    usage = resp.get("usage", {})
    out = "\n---\n\n### Provenance\n\n"
    out += _line("Model returned", resp.get("model_returned"))
    out += _line("API request id", resp.get("api_request_id"))
    out += _line("Attempt", f"{req.get('attempt')}"
                 + (f" (prior errors: {len(req.get('prior_attempt_errors') or [])})"
                    if req.get("prior_attempt_errors") else ""))
    out += _line("Effort / thinking", f"{req.get('effort')} / "
                 f"{(req.get('thinking') or {}).get('type')}")
    out += _line("Schema sha256", (req.get("output_schema_sha256") or "")[:16] + "...")
    out += _line("Scaffold sha256", (req.get("system_sha256") or "")[:16] + "...")
    out += _line("Context sha256", (log.get("retrieval", {})
                                     .get("context_block_sha256") or "")[:16] + "...")
    out += _line("Image set sha256", (iset.get("set_sha256") or "")[:16] + "...")
    out += _line("Magnifications", iset.get("magnifications_sent"))
    out += _line("Tile selection", iset.get("tile_selection_method"))
    out += _line("Tokens", f"in {usage.get('input_tokens')}, "
                 f"cache read {usage.get('cache_read_input_tokens', 0)}, "
                 f"out {usage.get('output_tokens')}")
    out += _line("Latency", f"{resp.get('latency_ms')} ms")
    out += ("\n*Research use only. Not a clinical report. Generated from "
            f"a protocol v{log.get('protocol_version')} case log.*\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", help="a case log json")
    ap.add_argument("--all", action="store_true",
                    help="render every current-protocol log")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    if args.all:
        outdir = pathlib.Path(args.outdir)
        n = 0
        for pathway in config.PATHWAYS:
            for path in sorted((pathlib.Path(config.LOG_ROOT) / pathway)
                               .glob("*.json")):
                log = json.loads(path.read_text())
                if log.get("protocol_version") != config.PROTOCOL_VERSION:
                    continue
                dest = outdir / pathway / (path.stem + ".md")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(render(log), encoding="utf-8")
                n += 1
        print(f"wrote {n} report(s) under {outdir}/")
        return

    if not args.log:
        ap.error("pass a log path or --all")
    print(render(json.loads(pathlib.Path(args.log).read_text())))


if __name__ == "__main__":
    main()
