# Testing

Four layers, none of which needs an API key.

| Layer | Command | What it proves | Cost |
|---|---|---|---|
| Unit | `python tests/test_smoke.py` | Magnification contract, MPATH-Dx v2.0 mapping, image determinism, scoring statistics | free |
| End-to-end | `python tests/test_e2e.py` | Runner → logger → verifier → scorer → report, against a scripted API double, with a transient failure, a refusal and a disagreeing replicate | free |
| Browser | `python tests/browser/test_app_browser.py` | Real Chromium against the real Streamlit app: real images, real clicks, real downloads | free (stubbed) |
| Live | `python tests/browser/test_app_browser.py --live` | The same, with real API calls | ~$0.16/run |

## Browser tests

```bash
pip install playwright && playwright install chromium

python tests/browser/test_app_browser.py              # local app, stubbed
python tests/browser/test_app_browser.py --repeat 20  # soak
python tests/browser/test_app_browser.py --url https://your-app.streamlit.app
python tests/browser/test_app_browser.py --live --yes # real API, costs money
python tests/browser/test_app_browser.py --headed     # watch it
```

Each run writes `_browser_test/runNNN/`: `results.json`, a screenshot of
each graded case, a screenshot of any failure, and the downloaded case
logs and synoptic reports. The run directory is wiped first, so a stale
failure screenshot can never sit in a passing run's artifacts.

### What the eleven checks cover

App loads with the right protocol and both tabs; the stub banner is
present exactly when stub mode is on; all three input modes are offered;
the URL and sample routes work without a file dialog; the sample case is
labelled as not-a-real-case; a wrong URL count is rejected; private and
link-local addresses are refused by the URL fetcher; a full grade round
trip on each pathway; and the downloaded log and report are well formed
(protocol and log version, four distinct images in the right order,
`schema_enforced`, no temperature, `structured_output` parsing, a
`[system, user]` envelope, and the report's diagnosis, provenance and
disclaimer sections).

### Two things that will bite you writing more of these

**Scope every selector to what is visible.** Streamlit keeps the
inactive tab's widgets in the DOM, so both pathways' "Fetch images" and
"Analyze …" buttons exist simultaneously. A plain
`get_by_role("button", …).first` resolves the hidden one and then times
out clicking an element with no bounding box — which reads as a broken
app and is really a broken selector. Use `driver.button()` and
`driver.choose_radio()`, which filter on `:visible`.

**Wait for the rerun, not the network.** Streamlit streams over a
WebSocket that never goes idle, so `wait_for_load_state("networkidle")`
hangs. `driver.wait_idle()` waits for `[data-testid="stStatusWidget"]`
to detach instead.

## Stub mode

`DERMPATH_STUB_API=1` makes `claude_transport.grade()` answer locally
with a fixed, schema-valid response instead of calling the API. Nothing
else changes: preprocessing, logging, the verifier's checks, the report
and the downloads all run for real.

It is opt-in and never a fallback — a missing key does not turn it on —
because a run that silently produced canned grades would be
indistinguishable from a real one afterwards. Every stubbed log records
`"model_returned": "stub"`, every stubbed field says so in its text, and
the app shows a red banner.

To test a deployed app in stub mode, add `DERMPATH_STUB_API = "1"` to
the host's secrets and reboot. **Remove it before real grading.**

## Sample case

`tests/fixtures/sample_case/` holds four images per pathway: centre
crops of one Creative Commons histopathology image at progressively
tighter framing, standing in for a whole-slide / 4x / 10x / 40x series.
`MANIFEST.json` carries the source, licence and crop fractions.

They exercise the pipeline. They are **not a real case**: the crops are
not true optical magnifications, there is no reference diagnosis, and a
grade produced from them is diagnostically meaningless. Never let one
into a results table.
