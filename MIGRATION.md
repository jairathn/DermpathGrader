# Getting off Replit

## What came across

`SquamousCancerGrader_export.zip` — 60 MB compressed, 103 MB expanded,
66 files. Built from `~/workspace` on the repl, SHA-256:

```
d0bfac4b88566084447c3f110df18ed763b70b9644b369fa924ff3eec4d28039
```

Contents:

- 15 Python modules (both grading pathways, the logging chain, the
  scoring and verification tooling)
- `chroma_db/` and `chroma_db_nevi/` — both vector stores, byte-identical
  to what the published runs used
- `attached_assets/` — 20 files, the source literature PDFs and test
  images
- `analysis_logs/` — prior CSCC and Nevus run output
- `run_manifest.json`, both ground-truth CSVs
- `pyproject.toml`, `uv.lock`, `.replit`, `replit.md`, `.streamlit/config.toml`

## What was left behind, deliberately

| Excluded | Size | Why |
|---|---|---|
| `.cache/` | 722 MB | uv/pip download cache, pure garbage |
| `.pythonlibs/` | 29 MB | installed site-packages, rebuilt locally |
| `.local/` | 8.2 MB | environment state |
| `__pycache__/` | 96 KB | compiled bytecode |
| `.git/` | 59 MB | see below |

Replit's own "Download as zip" would have tried to bundle all 916 MB,
most of it cache. That is why this was built by hand instead.

**On the git history:** four commits were in there, the most recent
being the verification-logging build. They were excluded per your call
to take code plus stores plus assets. If you want that provenance trail
for the manuscript record, it is still sitting on the repl and can be
pulled separately as a `git bundle`. Worth doing before the account
lapses for good.

**Not in any zip, by nature:** `ANTHROPIC_API_KEY`. Replit held it as a
platform Secret, which lives outside the filesystem. Nothing was lost,
but you will need to paste it into `.env`.

## Setup

Unzip the export, drop these scaffold files in alongside, then:

```bash
cd SquamousCancerGrader
./setup.sh          # creates .venv, installs deps, checks the stores
# put your key in .env
streamlit run app.py
```

`setup.sh` prefers `uv sync` so you get the exact pins from `uv.lock`.
It falls back to `requirements.txt`, which is approximate.

## Then start Claude Code

```bash
cd SquamousCancerGrader
git init && git add -A && git commit -m "Import from Replit"
claude
```

`CLAUDE.md` is already written and will be picked up automatically. It
carries the architecture map and, more importantly, the six landmines
worth knowing before anything gets changed. The `generate_examples.py`
one matters most if this feeds a supplement.

Commit before you let anything touch the tree. The vector stores are
not regenerable to their current state, so a clean baseline commit is
your only undo.

## Verify before you let the repl go

Do not delete the repl or let the account lapse until all of these pass
locally:

- [ ] `streamlit run app.py` loads and both pathways appear
- [ ] a CSCC case grades end to end against a real image
- [ ] a nevus case grades end to end
- [ ] `python verify_logging.py` runs and reports its check count
- [ ] retrieval distances match the reference values: CSCC subquery-1
      top-1 = 0.5462, nevus = 0.4066. If these drift, the stores did not
      come across intact and nothing downstream reproduces.
- [ ] `python join_and_score.py` reproduces the concordance you expect

That fifth item is the one that actually matters. Everything else can be
rebuilt; a shifted vector store quietly invalidates published numbers.

## On the suspended account

The export is done and the files are off the platform, so the billing
state no longer blocks the work. Two things are still worth doing while
the workspace loads at all:

1. Pull the `git bundle` if you want the history.
2. Check your other repls. The sidebar lists DermaAssistant,
   DermFlowEMR, BedBike, MobilityTracker, OncologyAdvisor, DermFlowAI
   and one with a mangled name. If any of those hold work that exists
   nowhere else, the same export method works and takes about two
   minutes each.
