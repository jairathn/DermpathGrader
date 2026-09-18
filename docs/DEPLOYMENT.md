# Deployment

## GitHub Pages will not work

GitHub Pages is a static file host. It serves HTML, CSS, JavaScript and
images, and it runs no server-side code. This application needs all three
of the things Pages cannot provide:

| Requirement | Why Pages cannot do it |
|---|---|
| A running Python process | Pages serves files; it does not execute anything |
| ChromaDB with persistent storage | Needs a filesystem and a live process |
| A server-side `ANTHROPIC_API_KEY` | See below |

The API key is the decisive one. On a static site every byte is public,
so a key shipped to the browser is readable by anyone who opens developer
tools, and it would be billed to your account by whoever finds it. There
is no configuration that fixes this: it is what "static hosting" means.

Pages **can** host the results, which is a different and worthwhile
thing. See "What can go on Pages" below.

## Before choosing a host: the slides

These are pathology slides. If they are patient specimens rather than a
public teaching set, where they are hosted stops being only an
engineering question and becomes an IRB and data-use question. A public
Streamlit URL is a public URL. Worth settling with whoever holds the data
agreement before picking from the list below, because it eliminates some
options outright.

Nothing in this repository uploads slides anywhere by itself. The
four JPEGs per case go to the Anthropic API when you grade a case; the
`.svs` files never leave the machine.

## Options that do work

Ordered by effort.

### 1. Streamlit Community Cloud — easiest

Free, deploys directly from a GitHub repo, and has a secrets manager so
the key stays server-side.

1. Push the repo to GitHub.
2. At share.streamlit.io, point it at the repo and `app.py`.
3. Add the key under Settings → Secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. `requirements.txt` is picked up automatically.

Caveats worth knowing up front:

- Apps are public by default. The free tier allows restricting viewers to
  named email addresses; do that before anything sensitive goes near it.
- Roughly 1 GB of RAM and no persistent disk between restarts. The vector
  stores must be committed to the repo, which they are — `chroma_db/` is
  1.3 MB. `analysis_logs/` written by the app will not survive a restart,
  so treat the hosted app as a demo and run batches locally.
- The ONNX embedding model downloads on first run (~80 MB), which makes
  the first request slow.

### 2. Hugging Face Spaces

Same shape, with private Spaces on the free tier and persistent storage
available. Use the Streamlit SDK or the Dockerfile in this repo. Key goes
in Settings → Variables and secrets.

### 3. Render, Railway or Fly.io

A real container with a persistent volume, so `analysis_logs/` survives
and batch runs can happen on the host. Use the included `Dockerfile`. A
few dollars a month; private by default.

### 4. Institutional cloud (Cloud Run, App Runner, Azure Container Apps)

The right answer if your institution has a cloud account with a BAA in
place. Same `Dockerfile`; mount a volume for `analysis_logs/`.

### 5. Run it locally

For a reader study this is often simply correct. No hosting, no data
leaving the machine, no exposure question to answer:

```bash
./setup.sh
streamlit run app.py --server.port 5000
```

To let colleagues on the same network reach it, bind to `0.0.0.0` and use
your machine's address.

## What can go on Pages

The **outputs** are static and publish cleanly: concordance tables,
confusion matrices, per-class breakdowns, retrieval provenance. Publish
`results/` as a static site and you get a shareable, citable URL for the
numbers without exposing a key or a slide.

Point Pages at `docs/` on the default branch, or add a workflow that
publishes `results/`. Ask and I will write the report generator that
turns `results/*.csv` into a static page.

## Docker

```bash
docker build -t dermpathgrader .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... \
  -v "$PWD/analysis_logs:/app/analysis_logs" dermpathgrader
```

The volume mount matters: without it, case logs vanish when the container
stops, and a log that does not survive is not a record.

## Never commit the key

`.env` is gitignored. Use the host's secrets manager, not a file in the
repo. If a key is ever pushed, rotate it at console.anthropic.com rather
than deleting the commit — the commit stays in the history and in every
clone.
