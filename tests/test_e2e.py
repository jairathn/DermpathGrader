"""End-to-end test of the whole grading chain without an API key.

    python tests/test_e2e.py

Builds a small synthetic study in a temporary directory, grades it
through the REAL batch runner (`run_tests.run_case`) with a scripted
stand-in for the Anthropic client, then runs the real verifier and the
real scorer over what was written. The script includes one transient
API failure (must retry and succeed), one refusal (must leave a failure
log, must not be scored), and one replicate that disagrees with the
others (must show up in replicate agreement).

Retrieval is replaced by a frozen, canned context block so the test
does not need the Chroma stores. Everything downstream of retrieval is
the production code path.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import anthropic  # noqa: E402
import httpx2 as httpx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import claude_transport  # noqa: E402
import config  # noqa: E402
from mock_anthropic import (FakeAnthropic, FakeMessage, FakeStopDetails,  # noqa: E402
                            FakeUsage, message_with_json)


class FakeClock:
    """Advances by a fixed step on every read, so latency is never 0."""

    def __init__(self, step_s: float = 0.25):
        self.now = 1_700_000_000.0
        self.step = step_s

    def time(self) -> float:
        self.now += self.step
        return self.now

    def sleep(self, _seconds: float) -> None:
        pass


@contextlib.contextmanager
def study_dir():
    """A temp cwd with config's relative paths pointing inside it."""
    with tempfile.TemporaryDirectory() as tmp:
        old = os.getcwd()
        os.chdir(tmp)
        try:
            pathlib.Path("data").mkdir()
            pathlib.Path(config.LOG_ROOT).mkdir()
            yield pathlib.Path(tmp)
        finally:
            os.chdir(old)


def make_tiles(pathway: str, case_id: str) -> dict[str, str]:
    d = pathlib.Path("tiles") / pathway / case_id
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for mag in config.MAGNIFICATIONS:
        p = d / f"{case_id}__{mag}.jpg"
        seed = abs(hash((case_id, mag))) % (2 ** 32)
        arr = np.random.default_rng(seed).integers(
            120, 255, (300, 400, 3), dtype="uint8")
        Image.fromarray(arr).save(p, quality=90)
        out[mag] = str(p)
    return out


NEVUS = [
    ("NEV-0001", "I", "dysplastic_nevus", "mild", "not_applicable", "not_applicable", None),
    ("NEV-0002", "II", "dysplastic_nevus", "severe", "not_applicable", "not_applicable", None),
    ("NEV-0003", "II", "melanoma", "not_applicable", "in_situ", "lentigo_maligna", None),
    ("NEV-0004", "III", "melanoma", "not_applicable", "invasive", "superficial_spreading", 0.5),
    ("NEV-0005", "IV", "melanoma", "not_applicable", "invasive", "nodular", 1.9),
]
CSCC = [("CSCC-0001", "Well Differentiated"),
        ("CSCC-0002", "Moderately Differentiated"),
        ("CSCC-0003", "Poorly Differentiated")]


def nevus_answer(cls, cat, grade, sub, hist, breslow):
    return {
        "mpath_dx_v2_class": cls, "lesion_category": cat,
        "specimen_adequacy": "adequate", "dysplasia_grade": grade,
        "melanoma_subtype": sub, "melanoma_histologic_subtype": hist,
        "breslow_estimate_mm": breslow, "ulceration_present": None,
        "mitoses_per_mm2": None, "confidence_level": "High",
        "differential_diagnosis": ["a", "b"],
        "recommended_ancillary_studies": ["PRAME"],
        "architectural_features": ["x"], "cytological_features": ["y"],
        "magnification_evidence": [{"magnification": m, "finding": "f"}
                                   for m in config.MAGNIFICATIONS],
        "grading_rationale": "r", "clinical_significance": "c",
    }


def cscc_answer(grade):
    broders = {"Well Differentiated": 1, "Moderately Differentiated": 2,
               "Poorly Differentiated": 4}[grade]
    return {
        "primary_grade": grade, "broders_grade": broders,
        "specimen_adequacy": "adequate", "histologic_subtype": "conventional",
        "depth_of_invasion": "reticular_dermis",
        "high_risk_features": (["poorly_differentiated"] if broders == 4
                               else ["none_identified"]),
        "confidence_level": "Medium", "keratinization_present": broders < 4,
        "atypia_level": "high" if broders == 4 else "minimal",
        "differential_diagnosis": ["KA"],
        "recommended_ancillary_studies": ["none"], "key_features": ["k"],
        "magnification_evidence": [{"magnification": m, "finding": "f"}
                                   for m in config.MAGNIFICATIONS],
        "additional_observations": "",
    }


def write_study() -> None:
    from build_case_registry import REGISTRY_FIELDS
    rows = []
    for cid, cls, cat, grade, sub, hist, breslow in NEVUS:
        t = make_tiles("Nevus", cid)
        rows.append({f: "" for f in REGISTRY_FIELDS} | {
            "case_id": cid, "pathway": "Nevus", "reference_stratum": cls,
            "lesion_category": cat, "dysplasia_grade": grade,
            "melanoma_subtype": sub, "melanoma_histologic_subtype": hist,
            "breslow_mm": "" if breslow is None else str(breslow),
            "tile_selection_method": "auto",
            **{f"tile_{m}": t[m] for m in config.MAGNIFICATIONS}})
    for cid, grade in CSCC:
        t = make_tiles("CSCC", cid)
        rows.append({f: "" for f in REGISTRY_FIELDS} | {
            "case_id": cid, "pathway": "CSCC",
            "reference_stratum": grade.split()[0].lower(),
            "tile_selection_method": "auto",
            **{f"tile_{m}": t[m] for m in config.MAGNIFICATIONS}})
    with open(config.CASE_REGISTRY, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    with open(config.GROUND_TRUTH["Nevus"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case_id", "reference_class", "lesion_category", "dysplasia_grade",
            "melanoma_subtype", "melanoma_histologic_subtype", "breslow_mm"])
        w.writeheader()
        for cid, cls, cat, grade, sub, hist, breslow in NEVUS:
            w.writerow({"case_id": cid, "reference_class": cls,
                        "lesion_category": cat, "dysplasia_grade": grade,
                        "melanoma_subtype": sub,
                        "melanoma_histologic_subtype": hist,
                        "breslow_mm": "" if breslow is None else breslow})
    with open(config.GROUND_TRUTH["CSCC"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "reference_grade"])
        w.writeheader()
        for cid, grade in CSCC:
            w.writerow({"case_id": cid, "reference_grade": grade})

    # Two readers; R2 disagrees with the reference on one nevus case.
    lines = ["reader_id,case_id,grade"]
    for rid in ("R1", "R2"):
        for cid, cls, *_ in NEVUS:
            lines.append(f"{rid},{cid},{'I' if (rid, cid) == ('R2', 'NEV-0002') else cls}")
    pathlib.Path("data/reader_grades_nevus.csv").write_text("\n".join(lines) + "\n")

    manifest = {
        "manifest_version": "2.0", "protocol_version": config.PROTOCOL_VERSION,
        "session_id": "e2e-session",
        "pathways": {pw: {"retrieval": {"context_block_sha256": ""}}
                     for pw in config.PATHWAYS},
    }
    pathlib.Path(config.RUN_MANIFEST).write_text(json.dumps(manifest))


def build_script() -> list:
    cached = FakeUsage(input_tokens=6000, output_tokens=900,
                       cache_read_input_tokens=12000)
    script = []
    for cid, cls, cat, grade, sub, hist, breslow in NEVUS:
        for rep in (1, 2, 3):
            answer = nevus_answer(cls, cat, grade, sub, hist, breslow)
            if cid == "NEV-0002" and rep == 3:
                answer["mpath_dx_v2_class"] = "I"       # disagreeing replicate
            if cid == "NEV-0004" and rep == 2:
                script.append(anthropic.APIConnectionError(   # transient
                    request=httpx.Request("POST", "https://api.test")))
            script.append(message_with_json(answer, usage=cached))
    script.append(FakeMessage(content=[], stop_reason="refusal",   # refusal
                              stop_details=FakeStopDetails(category="other")))
    for _, grade in CSCC:
        for _ in (1, 2, 3):
            script.append(message_with_json(cscc_answer(grade), usage=cached))
    return script


def test_end_to_end():
    import run_tests
    import verify_logging as V
    import join_and_score as S
    from grading_logger import CaseLogger

    clock = FakeClock()
    real_time = claude_transport.time
    claude_transport.time = clock                     # latency + backoff
    CaseLogger.LOG_ROOT = pathlib.Path(config.LOG_ROOT)

    try:
        with study_dir():
            write_study()
            manifest = json.loads(pathlib.Path(config.RUN_MANIFEST).read_text())
            client = FakeAnthropic(build_script())
            frozen = run_tests.FrozenRAG(
                "CANNED LITERATURE CONTEXT",
                [{"query": "q", "chunk_id": "doc_0", "rank": 1}])
            # The manifest carries the frozen context's hash, as it would
            # after a real make_manifest run.
            import hashlib
            ctx_sha = hashlib.sha256(b"CANNED LITERATURE CONTEXT").hexdigest()
            for pw in config.PATHWAYS:
                manifest["pathways"][pw]["retrieval"]["context_block_sha256"] = ctx_sha

            refused = 0
            for pathway in ("Nevus", "CSCC"):
                for row in run_tests.load_registry(pathway):
                    for _ in range(3):
                        run_tests.run_case(row, pathway, frozen, manifest,
                                           client=client)
                if pathway == "Nevus":
                    row = next(r for r in run_tests.load_registry("Nevus")
                               if r["case_id"] == "NEV-0005")
                    try:
                        run_tests.run_case(row, "Nevus", frozen, manifest,
                                           client=client)
                    except claude_transport.GradingRefused:
                        refused += 1
            assert refused == 1
            assert not client.messages.script, "script not fully consumed"

            logs = sorted(pathlib.Path(config.LOG_ROOT).glob("*/*.json"))
            assert len(logs) == 25, len(logs)

            # Retry recorded, refusal recorded
            retried = json.loads(pathlib.Path(
                config.LOG_ROOT, "Nevus", "NEV-0004__rep2.json").read_text())
            assert retried["request"]["attempt"] == 2
            assert len(retried["request"]["prior_attempt_errors"]) == 1
            assert retried["response"]["latency_ms"] > 0
            assert retried["response"]["usage"]["cache_read_input_tokens"] == 12000
            failed = json.loads(pathlib.Path(
                config.LOG_ROOT, "Nevus", "NEV-0005__rep4.json").read_text())
            assert failed["failure"]["error_class"] == "GradingRefused"
            assert failed["failure"]["category"] == "other"

            # Resume semantics: everything is complete, nothing planned
            work, summary = run_tests.plan_pathway(
                run_tests.load_registry("Nevus"), "Nevus", 3)
            assert work == [] and summary["complete_cases"] == 5
            assert summary["failed_replicates"] == 1

            # The real verifier, on every log, with the real manifest hashes
            V._checks.clear()
            for path in logs:
                log = json.loads(path.read_text())
                V.check_log(log, path.name, "e2e-session",
                            {pw: ctx_sha for pw in config.PATHWAYS})
            fails = [c for c in V._checks if c[1] == "FAIL"]
            assert not fails, "\n".join(f"{c[0]}: {c[2]}" for c in fails[:8])
            assert len(V._checks) > 2000

            # The real scorer
            nevus_logs, _ = S.partition_by_protocol(S.load_logs("Nevus"))
            assert len(nevus_logs) == 15         # refused log excluded
            nevus = S.score_nevus(nevus_logs, S.load_ground_truth("Nevus"))
            assert nevus["n_total"] == 15 and nevus["n_concordant"] == 14
            assert nevus["subtype_total"] == 9 and nevus["subtype_hits"] == 9
            assert nevus["melanoma"]["fn"] == 0 and nevus["melanoma"]["fp"] == 0
            rep = S.replicate_agreement(nevus_logs, "mpath_dx_v2_class",
                                        S.NEVUS_LABELS, S.normalize_class)
            assert rep["unanimous"] == 4 and rep["n_multi"] == 5
            assert rep["majority_by_case"]["NEV-0002"] == "II"
            readers = S.load_reader_grades("Nevus", S.normalize_class)
            assert set(readers) == {"R1", "R2"}
            table = S.reader_comparison(
                "Nevus", rep["majority_by_case"],
                {c: cls for c, cls, *_ in NEVUS}, S.NEVUS_LABELS, readers)
            by_name = {r["comparison"]: r for r in table}
            assert by_name["model_vs_reference"]["agreement"] == 1.0
            assert by_name["reader_R2_vs_reference"]["agreement"] == 0.8
            assert by_name["reader_consensus_vs_reference"]["agreement"] == 1.0

            cscc_logs, _ = S.partition_by_protocol(S.load_logs("CSCC"))
            cscc = S.score_cscc(cscc_logs, S.load_ground_truth("CSCC"))
            assert cscc["n_concordant"] == cscc["n_total"] == 9

            # Synoptic report renders for both a graded and a refused case
            import report
            text = report.render(retried)
            assert "Class III" in text and "Attempt:** 2" in text
            assert "No grade produced" in report.render(failed)
    finally:
        claude_transport.time = real_time


def main() -> None:
    try:
        test_end_to_end()
        print("  PASS  test_end_to_end")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  FAIL  test_end_to_end: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
