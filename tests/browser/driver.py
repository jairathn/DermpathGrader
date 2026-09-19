"""Playwright helpers for driving the Streamlit app.

Streamlit re-runs the whole script on every interaction and streams the
result over a WebSocket, so almost every failure in a naive browser test
is a race: the assertion runs against the previous render. `wait_idle`
is the fix and every action here goes through it.
"""

from __future__ import annotations

import contextlib
import os
import re
import pathlib
import socket
import subprocess
import time

from playwright.sync_api import Page, TimeoutError as PWTimeout

APP_READY_TIMEOUT_S = 180
ACTION_TIMEOUT_MS = 45_000

# Environments that ship Chromium pre-installed (this repo's CI sandbox
# among them) may carry a build that does not match the pip-installed
# Playwright's expected revision. Launching the bundled binary directly
# is the documented way through, and beats downloading a second copy.
_PREINSTALLED = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/opt/pw-browsers/chromium/chrome-linux/chrome",
]


def chromium_executable() -> str | None:
    """Path to a usable Chromium, or None to let Playwright pick."""
    override = os.environ.get("DERMPATH_CHROMIUM")
    if override:
        return override
    for candidate in _PREINSTALLED:
        if pathlib.Path(candidate).exists():
            return candidate
    for base in sorted(pathlib.Path("/opt/pw-browsers").glob("chromium-*"),
                       reverse=True):
        candidate = base / "chrome-linux" / "chrome"
        if candidate.exists():
            return str(candidate)
    return None


def launch_chromium(playwright, *, headless: bool = True):
    """Launch Chromium, preferring a pre-installed build when present."""
    kwargs = {"headless": headless,
              # --no-sandbox is required in the container this runs in;
              # it is a test browser against a local app, not a browser
              # handed untrusted pages.
              "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
    executable = chromium_executable()
    if executable:
        kwargs["executable_path"] = executable
    return playwright.chromium.launch(**kwargs)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def streamlit_server(*, stub: bool = True, port: int | None = None,
                     repo_root: pathlib.Path | None = None):
    """Run the real app in a subprocess and yield its base URL."""
    root = repo_root or pathlib.Path(__file__).resolve().parents[2]
    port = port or free_port()
    env = dict(os.environ)
    env["DERMPATH_STUB_API"] = "1" if stub else ""
    if stub:
        # Grading is stubbed, but the analyzers still construct a client.
        env.setdefault("ANTHROPIC_API_KEY", "sk-ant-stub-not-used")
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    process = subprocess.Popen(
        ["streamlit", "run", "app.py",
         "--server.port", str(port), "--server.address", "127.0.0.1",
         "--server.headless", "true"],
        cwd=root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + APP_READY_TIMEOUT_S
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"streamlit exited early:\n{process.stdout.read()[-2000:]}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("streamlit did not start in time")
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def wait_idle(page: Page, timeout_ms: int = ACTION_TIMEOUT_MS) -> None:
    """Block until Streamlit has finished re-running the script.

    Streamlit marks work in progress with [data-testid="stStatusWidget"].
    Waiting for it to detach is far more reliable than a sleep, and far
    more reliable than networkidle, which the app's WebSocket never
    reaches.
    """
    page.wait_for_timeout(250)          # let the rerun actually start
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]',
                               state="detached", timeout=timeout_ms)
    except PWTimeout:
        pass                             # never appeared, or already gone
    page.wait_for_timeout(250)


def open_app(page: Page, base_url: str) -> None:
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="stAppViewContainer"]',
                           timeout=ACTION_TIMEOUT_MS)
    wait_idle(page)


def select_tab(page: Page, label: str) -> None:
    page.get_by_role("tab", name=label).click()
    wait_idle(page)


def button(page: Page, label: str, *, exact: bool = False):
    """A *visible* button matching `label`.

    Visibility filtering is not optional here. Streamlit keeps the
    inactive tab's widgets in the DOM, so both pathways' "Fetch images"
    and "Analyze ... case" controls exist at once and a plain
    `get_by_role("button", ...).first` happily resolves to the hidden
    one, then times out clicking something with no bounding box. That
    failure looks like a broken app and is really a broken selector.
    """
    matcher = re.compile(rf"^\s*{re.escape(label)}\s*$") if exact else label
    return page.locator("button:visible").filter(has_text=matcher).first


def click(page: Page, label: str, *, exact: bool = False) -> None:
    button(page, label, exact=exact).click()
    wait_idle(page)


def choose_radio(page: Page, label: str) -> None:
    """Pick a radio option in the visible radio group.

    The <label> and its <input> both report a null bounding box; the
    text node inside is what actually takes the click.
    """
    group = page.locator('[data-testid="stRadioGroup"]:visible').first
    group.get_by_text(label, exact=True).click()
    wait_idle(page)


def textarea(page: Page):
    return page.locator("textarea:visible").first


def body_text(page: Page) -> str:
    return page.locator("body").inner_text()


def shot(page: Page, outdir: pathlib.Path, name: str) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return str(path)
