"""Join case logs to ground truth and compute concordance (protocol v2.0).

    python join_and_score.py [--outdir results]

Why this was rewritten rather than patched
------------------------------------------
The v1 scorer was built around two fields, `traditional_grade` (three
dysplasia tiers) and `mpath_grade` (a two-tier Low/High-Grade label).
Both are gone: the melanocytic label space is now four-way (melanoma is a
category, not the top of the dysplasia ladder) and the second field is a
real MPATH-Dx v2.0 class. There was no honest way to map the old arms
onto the new labels.

Scoring arms
------------
CSCC (one arm)
  CSCC_3tier                 well / moderately / poorly, exact match.

Melanocytic (four arms)
  Nevus_MPATH_v2_class       Class I/II/III/IV, exact match. The primary
                             arm: the strata are the classes, so every
                             case has exactly one correct answer and
                             there is no ambiguity to apportion.
  Nevus_management_binary    Class I versus Class II or above, i.e. the
                             re-excision decision. The arm that
                             corresponds to something happening to a
                             patient.
  Nevus_melanoma_detection   melanoma versus not, as sensitivity and
                             specificity. Scored on lesion_category, NOT
                             on class: Class II holds both high-grade
                             dysplasia and melanoma in situ, so the class
                             alone cannot establish melanoma.
  Nevus_insitu_vs_invasive   among cases the reference calls melanoma,
                             whether in situ and invasive are told apart.
                             This is the distinction that moves a case
                             between Class II and Classes III/IV, and it
                             is where a fixed-field read is most likely
                             to fail: depth of invasion is the hardest
                             thing to judge from four frames.

Nevus_internal_consistency is reported alongside these but is not a
concordance arm: it measures whether the model's own fields agree with
each other, which is a property of the output, not of the ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import config
import mpath_dx


# ── statistics ───────────────────────────────────────────────────────

def wilson(x: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval. Behaves at 0/n and n/n, unlike the normal
    approximation, which matters at 50 cases per stratum."""
    if n == 0:
        return (0.0, 0.0)
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    """Unweighted Cohen's kappa. Returns nan when it is undefined."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    ref_counts = Counter(a for a, _ in pairs)
    mod_counts = Counter(b for _, b in pairs)
    expected = sum((ref_counts[l] / n) * (mod_counts[l] / n) for l in labels)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def weighted_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    """Quadratic-weighted kappa, for the ordered grade scales.

    Ordinal grades deserve partial credit: calling a severe lesion
    moderate is a different error from calling it mild, and unweighted
    kappa treats both as equally wrong.
    """
    n = len(pairs)
    if n == 0:
        return float("nan")
    index = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    if k < 2:
        return float("nan")

    def w(i: int, j: int) -> float:
        return ((i - j) / (k - 1)) ** 2

    pairs = [(a, b) for a, b in pairs if a in index and b in index]
    n = len(pairs)
    if n == 0:
        return float("nan")

    observed = sum(w(index[a], index[b]) for a, b in pairs) / n
    ref_counts = Counter(a for a, _ in pairs)
    mod_counts = Counter(b for _, b in pairs)
    expected = sum(
        w(index[a], index[b]) * (ref_counts[a] / n) * (mod_counts[b] / n)
        for a in labels for b in labels)
    if expected == 0:
        return float("nan")
    return 1 - observed / expected


# ── loaders ──────────────────────────────────────────────────────────

def load_manifest() -> dict:
    p = pathlib.Path(config.RUN_MANIFEST)
    if not p.exists():
        sys.exit(f"{config.RUN_MANIFEST} not found. Run make_manifest.py.")
    return json.loads(p.read_text())


def load_ground_truth(pathway: str) -> dict[str, dict]:
    p = pathlib.Path(config.GROUND_TRUTH[pathway])
    if not p.exists():
        sys.exit(f"{p} not found.")
    with p.open(newline="", encoding="utf-8") as fh:
        return {row["case_id"]: row for row in csv.DictReader(fh)}


def load_logs(pathway: str) -> list[dict]:
    directory = pathlib.Path(config.LOG_ROOT) / pathway
    if not directory.exists():
        return []
    logs = []
    for path in sorted(directory.glob("*.json")):
        log = json.loads(path.read_text())
        log["_path"] = str(path)
        logs.append(log)
    return logs


def partition_by_protocol(logs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split current-protocol logs from everything else.

    Pooling a v1 single-image log with a v2 four-image log would compare
    two different experiments. They are separated here and the count of
    excluded logs is reported rather than dropped silently.
    """
    current, stale = [], []
    for log in logs:
        if log.get("protocol_version") == config.PROTOCOL_VERSION:
            current.append(log)
        else:
            stale.append(log)
    return current, stale


# ── normalisation ────────────────────────────────────────────────────

CSCC_LABELS = ["Well Differentiated", "Moderately Differentiated",
               "Poorly Differentiated"]
NEVUS_LABELS = list(config.NEVUS_STRATA)   # I, II, III, IV

_CSCC_ALIASES = {
    "well": "Well Differentiated",
    "well-differentiated": "Well Differentiated",
    "well differentiated": "Well Differentiated",
    "moderately": "Moderately Differentiated",
    "moderately-differentiated": "Moderately Differentiated",
    "moderately differentiated": "Moderately Differentiated",
    "moderate": "Moderately Differentiated",
    "poorly": "Poorly Differentiated",
    "poorly-differentiated": "Poorly Differentiated",
    "poorly differentiated": "Poorly Differentiated",
    "poor": "Poorly Differentiated",
}

# Legacy three-tier labels are NOT aliased to a class here on purpose.
# "moderate" spans Class I and Class II, so silently picking one would
# fabricate a reference answer. A registry still carrying three-tier
# labels fails loudly instead; convert it with
# mpath_dx.class_from_dysplasia_grade() and have a dermatopathologist
# resolve the ambiguous cases.


def normalize_cscc(grade: str) -> str:
    return _CSCC_ALIASES.get(str(grade).strip().lower(), str(grade).strip())


def normalize_class(value: str) -> str:
    """Normalise an MPATH-Dx class, or return "" if it is not one."""
    text = str(value or "").strip()
    if not text:
        return ""
    return mpath_dx._normalise(text) if mpath_dx.is_valid_class(text) else ""


def _float_or_none(value) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


# ── CSCC ─────────────────────────────────────────────────────────────

def score_cscc(logs: list[dict], gt: dict[str, dict]) -> dict:
    rows, pairs = [], []
    matrix: dict[tuple[str, str], int] = defaultdict(int)
    by_grade: dict[str, list[int]] = defaultdict(list)

    for log in logs:
        case_id = log["case_id"]
        reference = normalize_cscc(
            gt.get(case_id, {}).get("reference_grade", ""))
        model = normalize_cscc(
            log.get("parsing", {}).get("parsed", {}).get("primary_grade", ""))
        if not reference or not model:
            continue
        hit = int(reference == model)
        pairs.append((reference, model))
        matrix[(reference, model)] += 1
        by_grade[reference].append(hit)
        rows.append({
            "case_id": case_id,
            "replicate": log.get("replicate"),
            "reference_grade": reference,
            "model_grade": model,
            "concordant": hit,
            "confidence_level": log["parsing"]["parsed"].get(
                "confidence_level", ""),
            "magnifications_sent": "|".join(
                log.get("image_set", {}).get("magnifications_sent", [])),
        })

    n = len(pairs)
    nc = sum(1 for a, b in pairs if a == b)
    return {
        "rows": rows, "matrix": matrix, "labels": CSCC_LABELS,
        "n_total": n, "n_concordant": nc,
        "kappa": cohen_kappa(pairs, CSCC_LABELS),
        "weighted_kappa": weighted_kappa(pairs, CSCC_LABELS),
        "by_grade": {g: (sum(v), len(v)) for g, v in by_grade.items()},
    }


# ── melanocytic ──────────────────────────────────────────────────────

def score_nevus(logs: list[dict], gt: dict[str, dict]) -> dict:
    rows, pairs = [], []
    matrix: dict[tuple[str, str], int] = defaultdict(int)
    by_class: dict[str, list[int]] = defaultdict(list)

    mgmt_hits = mgmt_total = 0
    consistent = consistency_total = 0
    tp = fp = tn = fn = 0
    subtype_hits = subtype_total = 0
    breslow_pairs: list[tuple[float, float]] = []
    bad_reference: list[str] = []

    for log in logs:
        case_id = log["case_id"]
        gt_row = gt.get(case_id, {})
        parsed = log.get("parsing", {}).get("parsed", {})

        reference = normalize_class(gt_row.get("reference_class", ""))
        model = normalize_class(parsed.get("mpath_dx_v2_class", ""))
        if not reference:
            # Either no ground truth for this case, or a legacy label.
            if gt_row:
                bad_reference.append(case_id)
            continue
        if not model:
            continue

        # Arm 1: exact class
        hit = int(reference == model)
        pairs.append((reference, model))
        matrix[(reference, model)] += 1
        by_class[reference].append(hit)

        # Arm 2: re-excision decision
        mgmt_total += 1
        mgmt_hits += int(mpath_dx.requires_reexcision(model)
                         == mpath_dx.requires_reexcision(reference))

        # Arm 3: melanoma detection, from lesion_category. Class II holds
        # both high-grade dysplasia and melanoma in situ, so the class
        # cannot carry this.
        ref_category = (gt_row.get("lesion_category", "") or "").strip().lower()
        ref_subtype = (gt_row.get("melanoma_subtype", "") or "").strip().lower()
        ref_mel = (ref_category == "melanoma"
                   or ref_subtype in ("in_situ", "invasive"))
        model_category = (parsed.get("lesion_category", "") or "").strip()
        model_subtype = (parsed.get("melanoma_subtype", "") or "").strip()
        model_mel = model_category == "melanoma"

        if ref_mel and model_mel:
            tp += 1
        elif ref_mel and not model_mel:
            fn += 1
        elif not ref_mel and model_mel:
            fp += 1
        else:
            tn += 1

        # Arm 4: in situ vs invasive, among reference melanomas
        subtype_hit = None
        if ref_mel and ref_subtype in ("in_situ", "invasive"):
            subtype_total += 1
            subtype_hit = int(model_subtype == ref_subtype)
            subtype_hits += subtype_hit

        ref_breslow = _float_or_none(gt_row.get("breslow_mm", ""))
        model_breslow = _float_or_none(parsed.get("breslow_estimate_mm"))
        if ref_breslow is not None and model_breslow is not None:
            breslow_pairs.append((ref_breslow, model_breslow))

        flags = parsed.get("consistency_flags") or []
        consistency_total += 1
        consistent += int(not flags)

        rows.append({
            "case_id": case_id,
            "replicate": log.get("replicate"),
            "reference_class": reference,
            "model_class": model,
            "concordant": hit,
            "reference_lesion_category": ref_category,
            "model_lesion_category": model_category,
            "reference_melanoma_subtype": ref_subtype,
            "model_melanoma_subtype": model_subtype,
            "subtype_concordant": "" if subtype_hit is None else subtype_hit,
            "reference_histologic_subtype":
                gt_row.get("melanoma_histologic_subtype", ""),
            "model_histologic_subtype":
                parsed.get("melanoma_histologic_subtype", ""),
            "reference_breslow_mm": gt_row.get("breslow_mm", ""),
            "model_breslow_mm": parsed.get("breslow_estimate_mm", ""),
            "reference_dysplasia_grade": gt_row.get("dysplasia_grade", ""),
            "model_dysplasia_grade": parsed.get("dysplasia_grade", ""),
            "reference_reexcision":
                int(mpath_dx.requires_reexcision(reference)),
            "model_reexcision": int(mpath_dx.requires_reexcision(model)),
            "confidence_level": parsed.get("confidence_level", ""),
            "consistency_flags": "|".join(flags),
            "magnifications_sent": "|".join(
                log.get("image_set", {}).get("magnifications_sent", [])),
        })

    n = len(pairs)
    nc = sum(1 for a, b in pairs if a == b)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    breslow_mae = float("nan")
    breslow_pt1b_agree = float("nan")
    if breslow_pairs:
        breslow_mae = sum(abs(a - b) for a, b in breslow_pairs) / len(breslow_pairs)
        cut = mpath_dx.BRESLOW_PT1B_CUTOFF_MM
        breslow_pt1b_agree = sum(
            1 for a, b in breslow_pairs if (a < cut) == (b < cut)
        ) / len(breslow_pairs)

    return {
        "rows": rows, "matrix": matrix, "labels": NEVUS_LABELS,
        "n_total": n, "n_concordant": nc,
        "kappa": cohen_kappa(pairs, NEVUS_LABELS),
        "weighted_kappa": weighted_kappa(pairs, NEVUS_LABELS),
        "by_class": {c: (sum(v), len(v)) for c, v in by_class.items()},
        "mgmt_hits": mgmt_hits, "mgmt_total": mgmt_total,
        "melanoma": {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "sensitivity": sensitivity, "specificity": specificity},
        "subtype_hits": subtype_hits, "subtype_total": subtype_total,
        "breslow_n": len(breslow_pairs), "breslow_mae": breslow_mae,
        "breslow_pt1b_agreement": breslow_pt1b_agree,
        "consistent": consistent, "consistency_total": consistency_total,
        "bad_reference": bad_reference,
    }


# ── reporting ────────────────────────────────────────────────────────

def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: pathlib.Path, labels: list[str], matrix) -> None:
    rows = []
    for ref in labels:
        row = {"reference": ref}
        for model in labels:
            row[model] = matrix.get((ref, model), 0)
        rows.append(row)
    write_csv(path, rows)


def arm_row(name: str, hits: int, total: int, note: str = "") -> dict:
    low, high = wilson(hits, total)
    return {
        "arm": name, "n_concordant": hits, "n_total": total,
        "concordance": round(hits / total, 4) if total else "",
        "ci95_low": round(low, 4), "ci95_high": round(high, 4),
        "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    load_manifest()  # fail early if the run was never manifested
    arms: list[dict] = []
    notes: list[str] = []

    # CSCC
    cscc_logs, cscc_stale = partition_by_protocol(load_logs("CSCC"))
    if cscc_stale:
        notes.append(f"CSCC: excluded {len(cscc_stale)} log(s) from an "
                     f"earlier protocol version.")
    if cscc_logs:
        cscc = score_cscc(cscc_logs, load_ground_truth("CSCC"))
        write_csv(outdir / "cscc_cases.csv", cscc["rows"])
        write_confusion(outdir / "cscc_confusion.csv",
                        cscc["labels"], cscc["matrix"])
        arms.append(arm_row(
            "CSCC_3tier", cscc["n_concordant"], cscc["n_total"],
            f"kappa={cscc['kappa']:.3f} "
            f"quadratic-weighted={cscc['weighted_kappa']:.3f}"))
        for grade, (hits, total) in sorted(cscc["by_grade"].items()):
            arms.append(arm_row(f"CSCC_3tier::{grade}", hits, total,
                                "per-reference-grade breakdown"))

    # Melanocytic
    nevus_logs, nevus_stale = partition_by_protocol(load_logs("Nevus"))
    if nevus_stale:
        notes.append(f"Nevus: excluded {len(nevus_stale)} log(s) from an "
                     f"earlier protocol version.")
    if nevus_logs:
        nevus = score_nevus(nevus_logs, load_ground_truth("Nevus"))
        write_csv(outdir / "nevus_cases.csv", nevus["rows"])
        write_confusion(outdir / "nevus_confusion.csv",
                        nevus["labels"], nevus["matrix"])

        arms.append(arm_row(
            "Nevus_MPATH_v2_class", nevus["n_concordant"], nevus["n_total"],
            f"kappa={nevus['kappa']:.3f} "
            f"quadratic-weighted={nevus['weighted_kappa']:.3f}"))
        for mpath_class, (hits, total) in sorted(
                nevus["by_class"].items(), key=lambda kv: mpath_dx.class_rank(kv[0])):
            label = mpath_dx.CLASS_DEFINITIONS[mpath_class]["label"]
            arms.append(arm_row(
                f"Nevus_MPATH_v2_class::Class_{mpath_class}", hits, total,
                f"per-class breakdown ({label})"))

        arms.append(arm_row(
            "Nevus_management_binary", nevus["mgmt_hits"], nevus["mgmt_total"],
            "Class I vs Class II+ (re-excision decision)"))

        mel = nevus["melanoma"]
        sens_low, sens_high = wilson(mel["tp"], mel["tp"] + mel["fn"])
        spec_low, spec_high = wilson(mel["tn"], mel["tn"] + mel["fp"])
        arms.append({
            "arm": "Nevus_melanoma_detection::sensitivity",
            "n_concordant": mel["tp"], "n_total": mel["tp"] + mel["fn"],
            "concordance": round(mel["sensitivity"], 4)
            if mel["sensitivity"] == mel["sensitivity"] else "",
            "ci95_low": round(sens_low, 4), "ci95_high": round(sens_high, 4),
            "note": f"fn={mel['fn']} melanoma called non-melanoma; scored on "
                    f"lesion_category, not class",
        })
        arms.append({
            "arm": "Nevus_melanoma_detection::specificity",
            "n_concordant": mel["tn"], "n_total": mel["tn"] + mel["fp"],
            "concordance": round(mel["specificity"], 4)
            if mel["specificity"] == mel["specificity"] else "",
            "ci95_low": round(spec_low, 4), "ci95_high": round(spec_high, 4),
            "note": f"fp={mel['fp']} non-melanoma called melanoma",
        })

        arms.append(arm_row(
            "Nevus_insitu_vs_invasive",
            nevus["subtype_hits"], nevus["subtype_total"],
            "among reference melanomas; this is what moves a case between "
            "Class II and Classes III/IV"))

        if nevus["breslow_n"]:
            arms.append({
                "arm": "Nevus_breslow::pT1b_cutoff_agreement",
                "n_concordant": round(
                    nevus["breslow_pt1b_agreement"] * nevus["breslow_n"]),
                "n_total": nevus["breslow_n"],
                "concordance": round(nevus["breslow_pt1b_agreement"], 4),
                "ci95_low": "", "ci95_high": "",
                "note": f"agreement on the 0.8 mm cutoff; "
                        f"mean absolute error {nevus['breslow_mae']:.2f} mm",
            })

        arms.append(arm_row(
            "Nevus_internal_consistency",
            nevus["consistent"], nevus["consistency_total"],
            "model output self-agreement, not concordance with truth"))

        if nevus["bad_reference"]:
            notes.append(
                f"Nevus: {len(nevus['bad_reference'])} case(s) skipped - "
                f"reference_class is missing or is not an MPATH-Dx v2.0 "
                f"class. Legacy mild/moderate/severe labels are not "
                f"auto-converted, because 'moderate' spans Class I and "
                f"Class II and picking one would fabricate the reference. "
                f"First few: {', '.join(nevus['bad_reference'][:5])}")

    write_csv(outdir / "concordance.csv", arms)

    # Parser summary: under v2.0 any fallback at all is a defect.
    parser_rows = []
    for pathway in config.PATHWAYS:
        logs, _ = partition_by_protocol(load_logs(pathway))
        strategies = Counter(
            log.get("parsing", {}).get("strategy_used", "") for log in logs)
        fallbacks = sum(1 for log in logs
                        if log.get("parsing", {}).get("fallback_invoked"))
        truncated = sum(1 for log in logs
                        if log.get("response", {}).get(
                            "stop_reason") == "max_tokens")
        parser_rows.append({
            "pathway": pathway, "n_logs": len(logs),
            "fallback_invoked": fallbacks,
            "truncated_max_tokens": truncated,
            "strategies": json.dumps(dict(strategies)),
        })
    write_csv(outdir / "parser_summary.csv", parser_rows)

    print(f"\nWrote {outdir}/concordance.csv and per-case tables.\n")
    for row in arms:
        if "::" in row["arm"]:
            continue
        print(f"  {row['arm']:36} {row['n_concordant']:>4}/{row['n_total']:<4}"
              f"  {row['concordance']}  {row['note']}")
    for row in parser_rows:
        if row["fallback_invoked"] or row["truncated_max_tokens"]:
            print(f"\n  ! {row['pathway']}: {row['fallback_invoked']} "
                  f"fallback(s), {row['truncated_max_tokens']} truncated. "
                  f"Under v2.0 the schema is enforced, so either is a defect.")
    for note in notes:
        print(f"\n  ! {note}")


if __name__ == "__main__":
    main()
