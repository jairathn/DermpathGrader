"""Real-browser tests against the running Streamlit app.

    python tests/browser/test_app_browser.py                  # stubbed, free
    python tests/browser/test_app_browser.py --live           # real API, costs
    python tests/browser/test_app_browser.py --url https://... # a deployment
    python tests/browser/test_app_browser.py --repeat 20       # soak

Chromium drives the actual page: clicks the real controls, loads real
histopathology images, grades a case, and checks what comes back. It is
not a mock of the UI, it is the UI.

By default `DERMPATH_STUB_API=1`, so grading is answered locally and a
run costs nothing and needs no key. `--live` spends real money; the
runner makes you confirm.

Against a deployed URL the server is whatever you deployed, so `--url`
only runs the checks that do not depend on local files.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from playwright.sync_api import sync_playwright  # noqa: E402

import config  # noqa: E402
import driver  # noqa: E402

SAMPLE = ROOT / "tests" / "fixtures" / "sample_case"
RESULTS: list[dict] = []


# ── harness ──────────────────────────────────────────────────────────

class Failure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


@contextlib.contextmanager
def step(page, outdir: pathlib.Path, name: str):
    started = time.time()
    try:
        yield
    except Exception as exc:
        RESULTS.append({"test": name, "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.time() - started, 1),
                        "screenshot": driver.shot(page, outdir, f"FAIL_{name}")})
        print(f"  FAIL  {name}: {exc}")
    else:
        RESULTS.append({"test": name, "ok": True,
                        "seconds": round(time.time() - started, 1)})
        print(f"  pass  {name}  ({time.time() - started:.1f}s)")


# ── tests ────────────────────────────────────────────────────────────

def test_app_loads(page, outdir, **_):
    text = driver.body_text(page)
    check("Dermatopathology Grading System" in text, "title missing")
    check(f"v{config.PROTOCOL_VERSION}" in text,
          f"protocol v{config.PROTOCOL_VERSION} not shown")
    check(str(config.TARGET_N_TOTAL) in text,
          f"case total {config.TARGET_N_TOTAL} not shown")
    for tab in ("CSCC differentiation", "Melanocytic lesions"):
        check(page.get_by_role("tab", name=tab).count() > 0, f"tab {tab!r} missing")


def test_stub_banner_visible(page, outdir, stub, **_):
    text = driver.body_text(page)
    if stub:
        check("STUB MODE" in text,
              "stub mode is on but the page does not say so - a canned grade "
              "could be mistaken for a real one")
    else:
        check("STUB MODE" not in text, "stub banner shown in a live run")


def test_three_input_modes_offered(page, outdir, **_):
    driver.select_tab(page, "Melanocytic lesions")
    for mode in ("Upload files", "Image URLs", "Sample case"):
        check(page.locator('[data-testid="stRadioGroup"]:visible')
              .first.get_by_text(mode, exact=True).count() > 0,
              f"input mode {mode!r} missing")


def test_no_file_dialog_needed(page, outdir, **_):
    """The point of the URL and sample routes: drivable without a dialog."""
    driver.select_tab(page, "Melanocytic lesions")
    driver.choose_radio(page, "Image URLs")
    check(page.locator("textarea:visible").count() > 0, "URL textarea missing")
    check(driver.button(page, "Fetch images").count() > 0,
          "Fetch images button missing")
    driver.choose_radio(page, "Sample case")
    check(driver.button(page, "Load Nevus sample case").count() > 0,
          "sample case button missing")


def test_sample_case_warns_it_is_not_real(page, outdir, **_):
    driver.select_tab(page, "Melanocytic lesions")
    driver.choose_radio(page, "Sample case")
    text = driver.body_text(page)
    check("not a real case" in text.lower(),
          "sample case is not labelled as not-a-real-case")


def test_url_mode_rejects_wrong_count(page, outdir, **_):
    driver.select_tab(page, "Melanocytic lesions")
    driver.choose_radio(page, "Image URLs")
    driver.textarea(page).fill("https://example.org/only-one.jpg")
    driver.click(page, "Fetch images")
    text = driver.body_text(page)
    check(f"exactly {len(config.MAGNIFICATIONS)} URLs" in text,
          "one URL was not rejected with a count error")


def test_url_mode_blocks_internal_addresses(page, outdir, **_):
    """SSRF guard, exercised through the real UI."""
    driver.select_tab(page, "Melanocytic lesions")
    driver.choose_radio(page, "Image URLs")
    driver.textarea(page).fill("\n".join([
        "https://127.0.0.1/whole_slide.jpg",
        "https://10.0.0.1/4x.jpg",
        "https://169.254.169.254/10x.jpg",
        "https://192.168.1.1/40x.jpg"]))
    driver.click(page, "Fetch images")
    text = driver.body_text(page)
    check("not a public address" in text,
          "a private address was not refused by the URL fetcher")


def test_grade_sample_case(page, outdir, pathway, stub, **_):
    """The whole round trip: load four images, grade, read the result."""
    tab = "Melanocytic lesions" if pathway == "Nevus" else "CSCC differentiation"
    driver.select_tab(page, tab)

    driver.click(page, f"Initialize {pathway} RAG system")
    check("Ready" in driver.body_text(page),
          f"{pathway} RAG system did not report Ready")

    driver.choose_radio(page, "Sample case")
    driver.click(page, f"Load {pathway} sample case")
    text = driver.body_text(page)
    check("Sample case loaded" in text, "sample case did not load")
    check("Total payload" in text,
          "the prepared-image summary did not render, so preprocessing "
          "did not run")

    analyze = driver.button(page, f"Analyze {pathway} case")
    check(analyze.is_enabled(),
          "Analyze stayed disabled with four images loaded")
    analyze.click()
    driver.wait_idle(page, timeout_ms=180_000)

    text = driver.body_text(page)
    check("Analysis failed" not in text and "Could not reach" not in text,
          f"grading reported an error: "
          f"{[ln for ln in text.splitlines() if 'failed' in ln.lower()][:2]}")

    if pathway == "Nevus":
        check("MPATH-Dx v2.0 Class" in text, "no MPATH-Dx class in the result")
        check(any(f"Class {c}" in text for c in config.NEVUS_STRATA),
              "the class shown is not one of the study strata")
    else:
        check(any(g in text for g in ("Well Differentiated",
                                      "Moderately Differentiated",
                                      "Poorly Differentiated")),
              "no differentiation grade in the result")

    check("Logged to" in text, "no case log was written")
    check("Download case log" in text, "case log download missing")
    check("Download synoptic report" in text, "report download missing")
    driver.shot(page, outdir, f"graded_{pathway}")


def test_downloads_are_valid(page, outdir, pathway, **_):
    """Download the log and the report and check they are well formed."""
    tab = "Melanocytic lesions" if pathway == "Nevus" else "CSCC differentiation"
    driver.select_tab(page, tab)

    with page.expect_download(timeout=60_000) as caught:
        driver.button(page, "Download case log").click()
    log_path = outdir / f"{pathway}_log.json"
    caught.value.save_as(str(log_path))
    log = json.loads(log_path.read_text())

    check(log["protocol_version"] == config.PROTOCOL_VERSION,
          f"log protocol {log['protocol_version']!r} != "
          f"{config.PROTOCOL_VERSION!r}")
    check(log["log_version"] == config.LOG_VERSION, "log_version wrong")
    check(len(log["images"]) == len(config.MAGNIFICATIONS),
          f"log holds {len(log['images'])} images, expected "
          f"{len(config.MAGNIFICATIONS)}")
    check([i["magnification"] for i in log["images"]]
          == list(config.MAGNIFICATIONS), "magnification order wrong in log")
    check(len({i["sent_sha256"] for i in log["images"]})
          == len(config.MAGNIFICATIONS), "duplicate images in the log")
    check(log["request"]["schema_enforced"] is True, "schema not enforced")
    check(log["request"]["temperature"] is None, "temperature was sent")
    check(log["parsing"]["strategy_used"] == "structured_output",
          f"parser fell back to {log['parsing']['strategy_used']!r}")
    check(log["parsing"]["parsed"], "no parsed grade in the log")
    roles = [m["role"] for m in log["request"]["message_structure"]]
    check(roles == ["system", "user"], f"envelope roles {roles}")

    with page.expect_download(timeout=60_000) as caught:
        driver.button(page, "Download synoptic report").click()
    report_path = outdir / f"{pathway}_report.md"
    caught.value.save_as(str(report_path))
    report = report_path.read_text()
    check("## Diagnosis" in report, "report has no diagnosis section")
    check("### Provenance" in report, "report has no provenance section")
    check("Research use only" in report, "report is missing its disclaimer")


TESTS = [
    ("app_loads", test_app_loads, None),
    ("stub_banner", test_stub_banner_visible, None),
    ("three_input_modes", test_three_input_modes_offered, None),
    ("no_file_dialog_needed", test_no_file_dialog_needed, None),
    ("sample_case_disclaimed", test_sample_case_warns_it_is_not_real, None),
    ("url_wrong_count_rejected", test_url_mode_rejects_wrong_count, None),
    ("url_ssrf_blocked", test_url_mode_blocks_internal_addresses, None),
    ("grade_nevus", test_grade_sample_case, "Nevus"),
    ("downloads_nevus", test_downloads_are_valid, "Nevus"),
    ("grade_cscc", test_grade_sample_case, "CSCC"),
    ("downloads_cscc", test_downloads_are_valid, "CSCC"),
]


def run_once(base_url: str, outdir: pathlib.Path, *, stub: bool,
             headed: bool = False) -> None:
    with sync_playwright() as pw:
        browser = driver.launch_chromium(pw, headless=not headed)
        context = browser.new_context(viewport={"width": 1500, "height": 1200},
                                      accept_downloads=True)
        page = context.new_page()
        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}")
                if m.type == "error" else None)
        try:
            driver.open_app(page, base_url)
            for name, fn, pathway in TESTS:
                with step(page, outdir, name):
                    fn(page, outdir, pathway=pathway, stub=stub)
            if console:
                print(f"  note  {len(console)} browser console error(s); "
                      f"first: {console[0][:120]}")
        finally:
            context.close()
            browser.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="test a deployment instead of a local app")
    ap.add_argument("--live", action="store_true",
                    help="real API calls; costs money")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="skip the live-mode spend confirmation")
    ap.add_argument("--outdir", default="_browser_test")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    stub = not args.live
    failures = 0

    if args.live:
        # Two graded cases per run, at roughly 7-8 cents each with the
        # scaffold cached. Cheap once, not cheap in a loop, and the whole
        # point of stub mode is that you rarely need this.
        per_run = 2 * 0.08
        total = per_run * args.repeat
        print(f"LIVE MODE: {args.repeat} run(s) x 2 graded cases "
              f"~ ${total:.2f} of real API spend.")
        if not args.yes:
            if not sys.stdin.isatty():
                sys.exit("Refusing to spend money non-interactively. "
                         "Pass --yes if that is what you want.")
            if input("Proceed? [y/N] ").strip().lower() != "y":
                sys.exit("Aborted.")

    for iteration in range(1, args.repeat + 1):
        RESULTS.clear()
        run_dir = outdir / f"run{iteration:03d}"
        # Wipe first: a stale FAIL_*.png from an earlier run sitting in a
        # passing run's directory is worse than no screenshot at all.
        if run_dir.exists():
            shutil.rmtree(run_dir)
        label = "live API" if args.live else "stubbed"
        print(f"\n=== run {iteration}/{args.repeat} ({label}) ===")
        if args.url:
            run_once(args.url, run_dir, stub=stub, headed=args.headed)
        else:
            with driver.streamlit_server(stub=stub) as base:
                run_once(base, run_dir, stub=stub, headed=args.headed)

        passed = sum(1 for r in RESULTS if r["ok"])
        failures += len(RESULTS) - passed
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(RESULTS, indent=2))
        print(f"  {passed}/{len(RESULTS)} passed")

    print(f"\n{failures} failure(s) across {args.repeat} run(s). "
          f"Artifacts in {outdir}/")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
