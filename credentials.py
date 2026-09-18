"""Find the Anthropic API key, from whichever place this is running.

Import and call `ensure_api_key()` before anything constructs an
Anthropic client. It is idempotent and never overwrites a key that is
already in the environment.

Where the key may live, in the order tried
------------------------------------------
1. `ANTHROPIC_API_KEY` already in the environment. A key exported in the
   shell, or injected by a container host (Render, Fly, Cloud Run),
   always wins.
2. Streamlit secrets. On Streamlit Community Cloud the key is pasted
   into Settings -> Secrets, and this copies it into the environment so
   the analyzers and the batch tools find it the same way they would
   locally. Skipped silently when Streamlit is not installed or no
   script is running.
3. A `.env` file in the project root. This is the local-development
   path. `.env` is gitignored and must stay that way.

Where the key must NEVER live
-----------------------------
In the repository. Not in a source file, not in a committed `.env`, not
in a notebook output, not in a private repo either. GitHub scans pushes
for Anthropic keys and reports them to Anthropic, which revokes them
automatically - and the commit stays in the history and in every clone
that already pulled it. If a key is ever pushed, rotate it at
console.anthropic.com rather than trying to rewrite history.
"""

from __future__ import annotations

import os
import pathlib

ENV_VAR = "ANTHROPIC_API_KEY"
ENV_FILE = ".env"


def _from_streamlit_secrets() -> str | None:
    """The key from st.secrets, or None if unavailable.

    Wrapped defensively: importing streamlit outside a running script is
    fine, but touching st.secrets with no secrets file raises.
    """
    try:
        import streamlit as st
        value = st.secrets.get(ENV_VAR)          # type: ignore[union-attr]
    except Exception:
        return None
    return str(value) if value else None


def _parse_env_file(path: pathlib.Path) -> dict[str, str]:
    """Minimal KEY=value parser. No dependency, no shell evaluation.

    Handles `export KEY=value`, surrounding single or double quotes,
    blank lines and `#` comments. Deliberately does not expand `$VARS`:
    an API key is a literal, and expansion would be a way to smuggle
    shell semantics into a config file.
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            values[name] = value
    return values


def _from_env_file(start: pathlib.Path | None = None) -> str | None:
    root = start or pathlib.Path(__file__).resolve().parent
    return _parse_env_file(root / ENV_FILE).get(ENV_VAR) or None


def ensure_api_key(*, required: bool = False) -> str | None:
    """Put the key in os.environ if it can be found. Return it, or None.

    `required=True` raises RuntimeError with the three places to put it,
    which is more useful to a researcher than a bare KeyError from the
    SDK three frames later.
    """
    existing = os.environ.get(ENV_VAR)
    if existing:
        return existing

    for source in (_from_streamlit_secrets, _from_env_file):
        value = source()
        if value:
            os.environ[ENV_VAR] = value
            return value

    if required:
        raise RuntimeError(
            f"{ENV_VAR} is not set. Put it in one of:\n"
            f"  - Streamlit Community Cloud: Settings -> Secrets, as\n"
            f'        {ENV_VAR} = "sk-ant-..."\n'
            f"  - a local .env file in the project root (gitignored):\n"
            f"        {ENV_VAR}=sk-ant-...\n"
            f"  - the shell: export {ENV_VAR}=sk-ant-...\n"
            f"Never commit it to the repository."
        )
    return None


def describe_source() -> str:
    """Where the key came from, for doctor.py and the app sidebar."""
    if not os.environ.get(ENV_VAR):
        return "not found"
    if _from_streamlit_secrets():
        return "Streamlit secrets"
    if _from_env_file():
        return f"{ENV_FILE} file"
    return "environment variable"
