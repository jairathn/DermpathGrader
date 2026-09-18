"""
join_and_score.py
-----------------
Step 6 of the logging workflow.  Run after all cases are complete.

    python join_and_score.py [--pathway CSCC|Nevus|both]

Reads:
  run_manifest.json
  analysis_logs/{pathway}/{case_id}__rep{n}.json   (all replicates)
  ground_truth_cscc.csv / ground_truth_nevus.csv

Writes (results/ directory):
  case_results_{pathway}.csv
  confusion_cscc.csv
  confusion_nevus_traditional.csv
  confusion_nevus_mpath.csv
  concordance_wilson.csv
  repeat_agreement_{pathway}.csv
  retrieval_table_{pathway}.csv
  context_invariance.txt
  parser_fallback_summary.csv
"""

import argparse
import csv
import glob
import io
import json
import pathlib
import sys
from math import sqrt
from collections import defaultdict
from typing import Dict, List, Optional


OUT_DIR = pathlib.Path("results")

# ── Wilson confidence interval ────────────────────────────────────────────────

def wilson(x: int, n: int, z: float = 1.959963985):
    """Wilson score interval.  Returns (lower, upper)."""
    if n == 0:
        return 0.0, 1.0
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    p = pathlib.Path("run_manifest.json")
    if not p.exists():
        sys.exit("run_manifest.json not found.")
    return json.loads(p.read_text())


def load_ground_truth_cscc() -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with open("ground_truth_cscc.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["case_id"]] = row
    return rows


def load_ground_truth_nevus() -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with open("ground_truth_nevus.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["case_id"]] = row
    return rows


def load_logs(pathway: str) -> List[dict]:
    logs = []
    pattern = str(pathlib.Path("analysis_logs") / pathway / "*.json")
    for fp in sorted(glob.glob(pattern)):
        try:
            logs.append(json.loads(pathlib.Path(fp).read_text()))
        except Exception as e:
            print(f"  ⚠  Could not parse {fp}: {e}")
    return logs


# ── normalisation helpers ─────────────────────────────────────────────────────

# Map ground-truth spellings → canonical model spellings for comparison
CSCC_GRADE_MAP = {
    "Well-Differentiated":         "Well Differentiated",
    "Moderately-Differentiated":   "Moderately Differentiated",
    "Poorly-Differentiated":       "Poorly Differentiated",
}

NEVI_TRAD_MAP = {
    "Mild Dysplasia":     "Mild Dysplasia",
    "Moderate Dysplasia": "Moderate Dysplasia",
    "Severe Dysplasia":   "Severe Dysplasia",
}

NEVI_MPATH_MAP = {
    "Low-Grade Dysplasia":  "Low-Grade Dysplasia",
    "High-Grade Dysplasia": "High-Grade Dysplasia",
}

# Expected MPATH mapping from traditional grade (for internal consistency)
TRAD_TO_MPATH = {
    "Mild Dysplasia":     "Low-Grade Dysplasia",
    "Moderate Dysplasia": "Low-Grade Dysplasia",
    "Severe Dysplasia":   "High-Grade Dysplasia",
}


def normalize_cscc(grade: str) -> str:
    return CSCC_GRADE_MAP.get(grade, grade)


def concordant_cscc(ref_raw: str, model: str) -> bool:
    return normalize_cscc(ref_raw) == model


# ── context invariance ────────────────────────────────────────────────────────

def check_context_invariance(all_logs: Dict[str, List[dict]]) -> str:
    lines = []
    for pathway, logs in all_logs.items():
        hashes = {log["retrieval"]["context_block_sha256"] for log in logs
                  if log.get("retrieval", {}).get("context_block_sha256")}
        n = len(hashes)
        status = "✅ PASS" if n == 1 else f"❌ FAIL ({n} distinct hashes)"
        lines.append(f"{pathway}: {n} distinct context_block_sha256  {status}")
        if n != 1:
            for h in hashes:
                lines.append(f"   • {h}")
    return "\n".join(lines)


# ── CSCC scoring ──────────────────────────────────────────────────────────────

CSCC_LABELS = ["Well Differentiated", "Moderately Differentiated",
               "Poorly Differentiated"]


def score_cscc(logs: List[dict], gt: Dict[str, dict]) -> dict:
    # Rep-1 only for confusion / concordance
    rep1 = [l for l in logs if l.get("replicate") == 1]

    case_rows = []
    for log in logs:
        cid  = log["case_id"]
        rep  = log["replicate"]
        gt_row = gt.get(cid, {})
        ref_raw   = gt_row.get("reference_grade", "")
        ref_canon = normalize_cscc(ref_raw)
        parsed    = log.get("parsing", {}).get("parsed", {})
        model_grade = parsed.get("primary_grade", "")
        conc = concordant_cscc(ref_raw, model_grade) if ref_raw else None
        case_rows.append({
            "case_id":          cid,
            "replicate":        rep,
            "reference_grade":  ref_raw,
            "model_grade":      model_grade,
            "concordant":       conc,
            "confidence_level": parsed.get("confidence_level", ""),
            "keratinization":   parsed.get("keratinization_present", ""),
            "atypia_level":     parsed.get("atypia_level", ""),
            "key_features":     "; ".join(parsed.get("key_features", [])),
            "additional_obs":   parsed.get("additional_observations", ""),
            "api_request_id":   log.get("response", {}).get("api_request_id", ""),
        })

    # Confusion matrix (rep-1 only)
    conf = {r: {c: 0 for c in CSCC_LABELS} for r in CSCC_LABELS}
    for log in rep1:
        cid = log["case_id"]
        ref_raw = gt.get(cid, {}).get("reference_grade", "")
        ref_c   = normalize_cscc(ref_raw)
        model   = log.get("parsing", {}).get("parsed", {}).get("primary_grade", "")
        if ref_c in conf and model in conf[ref_c]:
            conf[ref_c][model] += 1

    # Overall concordance (rep-1)
    rep1_with_gt = [l for l in rep1 if gt.get(l["case_id"], {}).get("reference_grade")]
    n_concordant = sum(1 for l in rep1_with_gt
                       if concordant_cscc(gt[l["case_id"]]["reference_grade"],
                                          l.get("parsing", {}).get("parsed", {}).get("primary_grade", "")))
    n_total = len(rep1_with_gt)

    # Per-grade concordance
    grade_conc = {}
    for grade in CSCC_LABELS:
        grade_cases = [l for l in rep1_with_gt
                       if normalize_cscc(gt[l["case_id"]]["reference_grade"]) == grade]
        nc = sum(1 for l in grade_cases
                 if l.get("parsing", {}).get("parsed", {}).get("primary_grade", "") == grade)
        grade_conc[grade] = (nc, len(grade_cases))

    # Repeat agreement
    by_case: Dict[str, List[str]] = defaultdict(list)
    for log in logs:
        cid   = log["case_id"]
        grade = log.get("parsing", {}).get("parsed", {}).get("primary_grade", "")
        by_case[cid].append(grade)
    repeat_rows = []
    for cid, grades in by_case.items():
        all_same = len(set(grades)) == 1
        repeat_rows.append({"case_id": cid, "all_replicates_agree": all_same,
                            "grades": "; ".join(grades)})
    overall_agree = sum(1 for r in repeat_rows if r["all_replicates_agree"])

    return {
        "case_rows":    case_rows,
        "confusion":    conf,
        "n_concordant": n_concordant,
        "n_total":      n_total,
        "grade_conc":   grade_conc,
        "repeat_rows":  repeat_rows,
        "overall_agree": overall_agree,
    }


# ── Nevus scoring ─────────────────────────────────────────────────────────────

NEVI_TRAD_LABELS  = ["Mild Dysplasia", "Moderate Dysplasia", "Severe Dysplasia"]
NEVI_MPATH_LABELS = ["Low-Grade Dysplasia", "High-Grade Dysplasia"]


def score_nevus(logs: List[dict], gt: Dict[str, dict]) -> dict:
    rep1 = [l for l in logs if l.get("replicate") == 1]

    case_rows = []
    for log in logs:
        cid     = log["case_id"]
        rep     = log["replicate"]
        gt_row  = gt.get(cid, {})
        ref_trad  = gt_row.get("reference_traditional_grade", "")
        ref_mpath = gt_row.get("reference_mpath_grade", "")
        parsed    = log.get("parsing", {}).get("parsed", {})
        model_trad  = parsed.get("traditional_grade", "")
        model_mpath = parsed.get("mpath_grade", "")

        # Internal consistency: does model's own trad map to model's own mpath?
        expected_mpath = TRAD_TO_MPATH.get(model_trad, "")
        internally_consistent = (model_mpath == expected_mpath)

        case_rows.append({
            "case_id":                  cid,
            "replicate":                rep,
            "reference_traditional":    ref_trad,
            "model_traditional":        model_trad,
            "concordant_traditional":   (ref_trad == model_trad) if ref_trad else "",
            "reference_mpath":          ref_mpath,
            "model_mpath":              model_mpath,
            "concordant_mpath":         (ref_mpath == model_mpath) if ref_mpath else "",
            "internally_consistent":    internally_consistent,
            "confidence":               parsed.get("confidence_level", ""),
            "nuclear_count":            parsed.get("nuclear_abnormality_count", ""),
            "architectural_features":   "; ".join(parsed.get("architectural_features", [])),
            "cytological_features":     "; ".join(parsed.get("cytological_features", [])),
            "grading_rationale":        parsed.get("grading_rationale", ""),
            "clinical_significance":    parsed.get("clinical_significance", ""),
            "api_request_id":           log.get("response", {}).get("api_request_id", ""),
        })

    # Confusion matrices (rep-1 only)
    conf_trad  = {r: {c: 0 for c in NEVI_TRAD_LABELS}  for r in NEVI_TRAD_LABELS}
    conf_mpath = {r: {c: 0 for c in NEVI_MPATH_LABELS} for r in NEVI_MPATH_LABELS}

    for log in rep1:
        cid = log["case_id"]
        gt_row  = gt.get(cid, {})
        parsed  = log.get("parsing", {}).get("parsed", {})

        ref_t   = gt_row.get("reference_traditional_grade", "")
        model_t = parsed.get("traditional_grade", "")
        if ref_t in conf_trad and model_t in conf_trad.get(ref_t, {}):
            conf_trad[ref_t][model_t] += 1

        ref_m   = gt_row.get("reference_mpath_grade", "")
        model_m = parsed.get("mpath_grade", "")
        if ref_m in conf_mpath and model_m in conf_mpath.get(ref_m, {}):
            conf_mpath[ref_m][model_m] += 1

    # Concordance counts (rep-1)
    rep1_gt = [l for l in rep1 if gt.get(l["case_id"], {}).get("reference_traditional_grade")]
    nc_trad = sum(1 for l in rep1_gt
                  if l.get("parsing",{}).get("parsed",{}).get("traditional_grade","") ==
                     gt[l["case_id"]]["reference_traditional_grade"])
    nc_mpath = sum(1 for l in rep1_gt
                   if l.get("parsing",{}).get("parsed",{}).get("mpath_grade","") ==
                      gt[l["case_id"]].get("reference_mpath_grade",""))
    nc_consist = sum(1 for l in rep1_gt
                     if TRAD_TO_MPATH.get(
                         l.get("parsing",{}).get("parsed",{}).get("traditional_grade",""), "")
                        == l.get("parsing",{}).get("parsed",{}).get("mpath_grade",""))
    n_total = len(rep1_gt)

    # Per-grade traditional concordance
    trad_grade_conc = {}
    for grade in NEVI_TRAD_LABELS:
        glist = [l for l in rep1_gt
                 if gt[l["case_id"]]["reference_traditional_grade"] == grade]
        nc = sum(1 for l in glist
                 if l.get("parsing",{}).get("parsed",{}).get("traditional_grade","") == grade)
        trad_grade_conc[grade] = (nc, len(glist))

    # Per-grade MPATH concordance
    mpath_grade_conc = {}
    for grade in NEVI_MPATH_LABELS:
        glist = [l for l in rep1_gt
                 if gt[l["case_id"]].get("reference_mpath_grade","") == grade]
        nc = sum(1 for l in glist
                 if l.get("parsing",{}).get("parsed",{}).get("mpath_grade","") == grade)
        mpath_grade_conc[grade] = (nc, len(glist))

    # Repeat agreement
    by_case_t: Dict[str, List[str]] = defaultdict(list)
    by_case_m: Dict[str, List[str]] = defaultdict(list)
    for log in logs:
        cid = log["case_id"]
        by_case_t[cid].append(log.get("parsing",{}).get("parsed",{}).get("traditional_grade",""))
        by_case_m[cid].append(log.get("parsing",{}).get("parsed",{}).get("mpath_grade",""))

    repeat_rows = []
    for cid in by_case_t:
        gt_trad  = by_case_t[cid]
        gt_mpath = by_case_m[cid]
        repeat_rows.append({
            "case_id":            cid,
            "traditional_agree":  len(set(gt_trad))  == 1,
            "traditional_grades": "; ".join(gt_trad),
            "mpath_agree":        len(set(gt_mpath)) == 1,
            "mpath_grades":       "; ".join(gt_mpath),
        })
    trad_agree  = sum(1 for r in repeat_rows if r["traditional_agree"])
    mpath_agree = sum(1 for r in repeat_rows if r["mpath_agree"])

    return {
        "case_rows":       case_rows,
        "conf_trad":       conf_trad,
        "conf_mpath":      conf_mpath,
        "nc_trad":         nc_trad,
        "nc_mpath":        nc_mpath,
        "nc_consist":      nc_consist,
        "n_total":         n_total,
        "trad_grade_conc": trad_grade_conc,
        "mpath_grade_conc":mpath_grade_conc,
        "repeat_rows":     repeat_rows,
        "trad_agree":      trad_agree,
        "mpath_agree":     mpath_agree,
    }


# ── retrieval table ───────────────────────────────────────────────────────────

def build_retrieval_table(manifest: dict, pathway: str) -> List[dict]:
    results = manifest["pathways"][pathway]["retrieval"]["results"]
    # Group by subquery, compute min/max distance across top 5
    from collections import defaultdict
    by_sq: Dict[int, List[dict]] = defaultdict(list)
    for r in results:
        by_sq[r["subquery_n"]].append(r)

    rows = []
    for sq_n in sorted(by_sq):
        entries = by_sq[sq_n]
        distances = [e["distance"] for e in entries]
        sources   = list({e["source"] for e in entries})
        chunk_ids = [e["chunk_id"] for e in entries]
        rows.append({
            "subquery_n":    sq_n,
            "subquery":      entries[0]["subquery"],
            "min_distance":  round(min(distances), 4),
            "max_distance":  round(max(distances), 4),
            "source_docs":   "; ".join(sorted(sources)),
            "chunk_ids":     "; ".join(chunk_ids),
            "content_type":  "",   # filled by hand per spec
        })
    return rows


# ── writers ───────────────────────────────────────────────────────────────────

def write_csv(filepath: pathlib.Path, rows: List[dict]):
    if not rows:
        filepath.write_text("(no data)\n", encoding="utf-8")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_confusion_csv(filepath: pathlib.Path, labels: List[str], matrix: dict):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["ref \\ model"] + labels
        w.writerow(header)
        for ref in labels:
            row = [ref] + [matrix.get(ref, {}).get(col, 0) for col in labels]
            w.writerow(row)


def write_concordance_csv(filepath: pathlib.Path,
                          cscc: Optional[dict], nevus: Optional[dict]):
    rows = []

    def add_arm(arm_name, nc, n, grade_conc):
        lo, hi = wilson(nc, n)
        rows.append({
            "arm":           arm_name,
            "grade":         "Overall",
            "concordant":    nc,
            "total":         n,
            "proportion":    round(nc / n, 4) if n else "",
            "wilson_lo_95":  round(lo, 4),
            "wilson_hi_95":  round(hi, 4),
        })
        for grade, (gnc, gn) in grade_conc.items():
            glo, ghi = wilson(gnc, gn)
            rows.append({
                "arm":           arm_name,
                "grade":         grade,
                "concordant":    gnc,
                "total":         gn,
                "proportion":    round(gnc / gn, 4) if gn else "",
                "wilson_lo_95":  round(glo, 4),
                "wilson_hi_95":  round(ghi, 4),
            })

    if cscc:
        add_arm("CSCC_3tier", cscc["n_concordant"], cscc["n_total"],
                cscc["grade_conc"])

    if nevus:
        add_arm("Nevus_traditional_3tier", nevus["nc_trad"], nevus["n_total"],
                nevus["trad_grade_conc"])
        add_arm("Nevus_MPATH_2tier",       nevus["nc_mpath"], nevus["n_total"],
                nevus["mpath_grade_conc"])
        # Internal consistency arm (no grade breakdown)
        lo, hi = wilson(nevus["nc_consist"], nevus["n_total"])
        rows.append({
            "arm":           "Nevus_internal_consistency",
            "grade":         "Overall",
            "concordant":    nevus["nc_consist"],
            "total":         nevus["n_total"],
            "proportion":    round(nevus["nc_consist"] / nevus["n_total"], 4)
                             if nevus["n_total"] else "",
            "wilson_lo_95":  round(lo, 4),
            "wilson_hi_95":  round(hi, 4),
        })

    write_csv(filepath, rows)


def write_parser_summary(filepath: pathlib.Path,
                         all_logs: Dict[str, List[dict]]):
    rows = []
    for pathway, logs in all_logs.items():
        by_strategy: Dict[str, List[str]] = defaultdict(list)
        for log in logs:
            strat = log.get("parsing", {}).get("strategy_used", "unknown")
            by_strategy[strat].append(log["case_id"])
        for strat, cases in by_strategy.items():
            rows.append({
                "pathway":         pathway,
                "strategy_used":   strat,
                "count":           len(cases),
                "case_ids":        "; ".join(sorted(set(cases))),
            })
    write_csv(filepath, rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathway", choices=["CSCC", "Nevus", "both"],
                        default="both")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    manifest = load_manifest()
    run_pathways = (["CSCC", "Nevus"] if args.pathway == "both"
                    else [args.pathway])

    cscc_result  = None
    nevus_result = None
    all_logs: Dict[str, List[dict]] = {}

    for pathway in run_pathways:
        logs = load_logs(pathway)
        if not logs:
            print(f"  ⚠  No logs found for {pathway} — skipping.")
            continue
        all_logs[pathway] = logs
        print(f"{pathway}: {len(logs)} log files loaded.")

        if pathway == "CSCC":
            gt = load_ground_truth_cscc()
            r  = score_cscc(logs, gt)
            cscc_result = r

            write_csv(OUT_DIR / "case_results_CSCC.csv", r["case_rows"])
            write_confusion_csv(OUT_DIR / "confusion_cscc.csv",
                                CSCC_LABELS, r["confusion"])

            repeat = r["repeat_rows"] + [{
                "case_id": "_OVERALL",
                "all_replicates_agree": r["overall_agree"],
                "grades": f"{r['overall_agree']}/{len(r['repeat_rows'])} cases fully agree",
            }]
            write_csv(OUT_DIR / "repeat_agreement_CSCC.csv", repeat)

        else:  # Nevus
            gt = load_ground_truth_nevus()
            r  = score_nevus(logs, gt)
            nevus_result = r

            write_csv(OUT_DIR / "case_results_Nevus.csv", r["case_rows"])
            write_confusion_csv(OUT_DIR / "confusion_nevus_traditional.csv",
                                NEVI_TRAD_LABELS, r["conf_trad"])
            write_confusion_csv(OUT_DIR / "confusion_nevus_mpath.csv",
                                NEVI_MPATH_LABELS, r["conf_mpath"])

            repeat = r["repeat_rows"] + [{
                "case_id":            "_OVERALL",
                "traditional_agree":  r["trad_agree"],
                "traditional_grades": f"{r['trad_agree']}/{len(r['repeat_rows'])} cases",
                "mpath_agree":        r["mpath_agree"],
                "mpath_grades":       f"{r['mpath_agree']}/{len(r['repeat_rows'])} cases",
            }]
            write_csv(OUT_DIR / "repeat_agreement_Nevus.csv", repeat)

        # Retrieval table
        tbl = build_retrieval_table(manifest, pathway)
        write_csv(OUT_DIR / f"retrieval_table_{pathway}.csv", tbl)

    # Concordance (all arms in one file)
    write_concordance_csv(OUT_DIR / "concordance_wilson.csv",
                          cscc_result, nevus_result)

    # Context invariance
    inv_text = check_context_invariance(all_logs)
    (OUT_DIR / "context_invariance.txt").write_text(inv_text, encoding="utf-8")
    print("\nContext invariance:")
    print(inv_text)

    # Parser fallback summary
    write_parser_summary(OUT_DIR / "parser_fallback_summary.csv", all_logs)

    print(f"\nAll outputs written to {OUT_DIR}/")
    print("Fill retrieval_table_*.csv  content_type column by hand before submitting.")

    # Quick self-check on Wilson
    lo20, hi20 = wilson(20, 20)
    lo54, hi54 = wilson(54, 60)
    print(f"\nWilson self-check: wilson(20,20)=({lo20:.4f},{hi20:.4f})  "
          f"expect (0.8389,1.0000)")
    print(f"                   wilson(54,60)=({lo54:.4f},{hi54:.4f})  "
          f"expect (0.7985,0.9534)")


if __name__ == "__main__":
    main()
