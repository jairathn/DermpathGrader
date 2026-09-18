"""SQLite shim for hosts that ship an old system SQLite.

Import this BEFORE chromadb, anywhere chromadb is used.

Why this exists
---------------
ChromaDB requires SQLite 3.35 or newer. Streamlit Community Cloud (and a
number of other managed Python hosts) run a Debian image whose system
SQLite predates that, so `import chromadb` dies at startup with

    RuntimeError: Your system has an unsupported version of sqlite3.
    Chroma requires sqlite3 >= 3.35.0

You cannot apt-get a newer SQLite on those hosts. The accepted workaround
is to install `pysqlite3-binary`, which bundles a current SQLite as a
wheel, and swap it into `sys.modules` under the name `sqlite3` before
anything imports chromadb.

This module does that, and only that, and only when it is actually
needed: if the system SQLite is already new enough, or pysqlite3 is not
installed, it does nothing and local development is unaffected.
"""

from __future__ import annotations

import sqlite3 as _system_sqlite
import sys

MINIMUM = (3, 35, 0)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def ensure_modern_sqlite() -> str:
    """Swap in pysqlite3 if the system SQLite is too old for ChromaDB.

    Returns a short status string, useful in a startup diagnostic.
    """
    current = _version_tuple(_system_sqlite.sqlite_version)
    if current >= MINIMUM:
        return f"system sqlite3 {_system_sqlite.sqlite_version} (ok)"

    try:
        __import__("pysqlite3")
    except ImportError:
        return (f"system sqlite3 {_system_sqlite.sqlite_version} is below "
                f"{'.'.join(map(str, MINIMUM))} and pysqlite3-binary is not "
                f"installed; chromadb will fail to import")

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    sys.modules["sqlite3.dbapi2"] = sys.modules["sqlite3"].dbapi2
    return (f"swapped in pysqlite3 "
            f"{sys.modules['sqlite3'].sqlite_version} (system was "
            f"{_system_sqlite.sqlite_version})")


STATUS = ensure_modern_sqlite()
