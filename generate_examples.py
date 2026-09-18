"""
generate_examples.py
--------------------
Produces representative examples of:
  1. Query patterns sent to ChromaDB
  2. Literature text snippets actually retrieved
  3. LLM prompt structure (with context injected)
  4. How to log LLM reasoning for concordant vs non-concordant cases

Run with:  python generate_examples.py
No API key needed – it only touches ChromaDB and prints examples.
To also run a live LLM call, set ANTHROPIC_API_KEY and use --live flag.
"""

import pathlib
import csv
import json
import textwrap
import argparse
import chromadb

# ── helpers ──────────────────────────────────────────────────────────────────

DIVIDER   = "=" * 78
SUBDIV    = "-" * 78
WRAP_WIDTH = 78


def box(title: str) -> str:
    return f"\n{DIVIDER}\n  {title}\n{DIVIDER}"


def wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=WRAP_WIDTH, initial_indent=prefix,
                         subsequent_indent=prefix)


# ── 1. QUERY PATTERNS ────────────────────────────────────────────────────────

SCC_QUERIES = [
    "well differentiated squamous cell carcinoma characteristics keratinization",
    "moderately differentiated squamous cell carcinoma features",
    "poorly differentiated squamous cell carcinoma criteria atypia",
    "keratin pearls horn cysts squamous cell carcinoma grading",
    "cellular atypia pleomorphism squamous cell carcinoma differentiation",
    "Broders grading system squamous cell carcinoma",
    "WHO grading squamous cell carcinoma differentiation",
]

NEVI_QUERIES = [
    "mild dysplasia atypical melanocytic nevi characteristics",
    "moderate dysplasia atypical melanocytic nevi features",
    "severe dysplasia atypical melanocytic nevi criteria",
    "low grade dysplasia melanocytic lesions",
    "high grade dysplasia melanocytic lesions",
    "nuclear atypia melanocytic nevi grading",
    "architectural disorder melanocytic nevi",
    "junctional melanocytic hyperplasia",
    "MPATH-Dx classification melanocytic lesions",
    "dysplastic nevus histologic criteria",
    "melanocytic atypia grading system",
    "WHO classification dysplastic nevi",
]


def print_query_patterns():
    print(box("SECTION 1 — QUERY PATTERNS SENT TO ChromaDB"))

    print("\n  ── SCC System (rag_system.py → get_grading_criteria) ──")
    for i, q in enumerate(SCC_QUERIES, 1):
        print(f"   [{i:02d}] {q}")

    print("\n  ── Nevi System (nevi_rag_system.py → get_grading_criteria) ──")
    for i, q in enumerate(NEVI_QUERIES, 1):
        print(f"   [{i:02d}] {q}")

    print("""
  How these queries work
  ──────────────────────
  Each query string is embedded by ChromaDB's default embedding function
  (sentence-transformers/all-MiniLM-L6-v2) and compared via cosine similarity
  against the 72 SCC / 294 Nevi document chunks stored in the vector DB.
  The top n_results=5 nearest chunks per query are returned and concatenated
  to form the context block injected into the LLM prompt.
""")


# ── 2. RETRIEVED SNIPPETS ────────────────────────────────────────────────────

def query_chroma(db_path: str, collection_name: str,
                 queries: list, n_results: int = 3) -> list:
    """Return list of {query, results} dicts."""
    client = chromadb.PersistentClient(path=db_path)
    col    = client.get_collection(collection_name)

    output = []
    for q in queries:
        r = col.query(query_texts=[q], n_results=n_results)
        snippets = []
        for i, doc in enumerate(r["documents"][0]):
            snippets.append({
                "rank":     i + 1,
                "distance": round(r["distances"][0][i], 4),
                "source":   r["metadatas"][0][i].get("source", "?"),
                "page":     r["metadatas"][0][i].get("page", "?"),
                "text":     doc.strip(),
            })
        output.append({"query": q, "results": snippets})
    return output


def print_retrieved_snippets():
    print(box("SECTION 2 — LITERATURE TEXT SNIPPETS RETRIEVED FROM ChromaDB"))

    # ── SCC: show first 3 queries ──
    print("\n  ── SCC Collection (chroma_db / scc_grading_literature) ──")
    scc_data = query_chroma("./chroma_db", "scc_grading_literature",
                            SCC_QUERIES[:3], n_results=2)

    for entry in scc_data:
        print(f"\n  Query: \"{entry['query']}\"")
        for s in entry["results"]:
            src_short = s["source"].replace("attached_assets/", "")
            print(f"\n    [Rank {s['rank']} | distance={s['distance']} | "
                  f"{src_short} p.{s['page']}]")
            # Print first 400 chars of snippet
            snippet_preview = s["text"][:400].replace("\n", " ")
            print(wrap(snippet_preview + ("…" if len(s["text"]) > 400 else "")))

    # ── Nevi: show first 3 queries ──
    print("\n\n  ── Nevi Collection (chroma_db_nevi / nevi_grading_literature) ──")
    nevi_data = query_chroma("./chroma_db_nevi", "nevi_grading_literature",
                             NEVI_QUERIES[:3], n_results=2)

    for entry in nevi_data:
        print(f"\n  Query: \"{entry['query']}\"")
        for s in entry["results"]:
            src_short = s["source"].replace("attached_assets/", "")
            print(f"\n    [Rank {s['rank']} | distance={s['distance']} | "
                  f"{src_short} p.{s['page']}]")
            snippet_preview = s["text"][:400].replace("\n", " ")
            print(wrap(snippet_preview + ("…" if len(s["text"]) > 400 else "")))

    return scc_data, nevi_data


# ── 3. LLM PROMPT STRUCTURE ──────────────────────────────────────────────────

SCC_PROMPT_TEMPLATE = """
You are an expert dermatopathologist analyzing a histopathology image of
cutaneous squamous cell carcinoma (CSCC) for differentiation grading.

Based on the following medical literature context from authoritative sources:

{context}

ANALYZE the provided histopathology image for these features:
  - Keratinization: keratin pearls, horn cysts, intracellular keratin
  - Cellular atypia: pleomorphism, nuclear abnormalities
  - Tumor architecture: organized vs infiltrative/disorganized
  - Squamous maturation: gradient from basal to keratinized cells
  - Mitotic activity: low, moderate, or high

GRADING CRITERIA:
  - Well Differentiated    : abundant keratinization, keratin pearls, minimal atypia
  - Moderately Differentiated: some keratinization, moderate atypia
  - Poorly Differentiated  : minimal/absent keratinization, high atypia, basaloid

YOU MUST PROVIDE A GRADE. Respond with a JSON object:
```json
{{
  "primary_grade": "Well Differentiated|Moderately Differentiated|Poorly Differentiated",
  "confidence_level": "High|Medium|Low",
  "keratinization_present": true|false,
  "atypia_level": "minimal|moderate|high",
  "key_features": ["feature 1", "feature 2", "feature 3"],
  "additional_observations": "..."
}}
```
"""

NEVI_PROMPT_TEMPLATE = """
Expert Dermatopathologist. Analyze the image using the provided context:
{context}

GRADING CRITERIA:
  - Severe Dysplasia  : 3+ nuclear abnormalities OR any architectural high-risk feature
  - Moderate Dysplasia: Nuclear size 1.5x, some pleomorphism/chromatin clumping
  - Mild Dysplasia    : Nuclear size 1x, minimal atypia, symmetric

OUTPUT JSON ONLY:
{{
  "traditional_grade": "Mild Dysplasia|Moderate Dysplasia|Severe Dysplasia",
  "mpath_grade": "Low-Grade Dysplasia|High-Grade Dysplasia",
  "confidence_level": "High|Medium|Low",
  "nuclear_abnormality_count": number,
  "architectural_features": [...],
  "cytological_features": [...],
  "grading_rationale": "...",
  "clinical_significance": "..."
}}
"""


def print_prompt_structure(scc_data: list, nevi_data: list):
    print(box("SECTION 3 — LLM PROMPT STRUCTURE (with real retrieved context)"))

    # Build a short real context from the first SCC query result
    scc_context_sample = "\n\n".join(
        s["text"][:300] for s in scc_data[0]["results"]
    )
    filled_scc = SCC_PROMPT_TEMPLATE.format(context=scc_context_sample[:600] + "…")

    print("\n  ── SCC Prompt sent to Claude claude-opus-4-5 ──")
    print(SUBDIV)
    print(filled_scc)
    print(SUBDIV)

    nevi_context_sample = "\n\n".join(
        s["text"][:300] for s in nevi_data[0]["results"]
    )
    filled_nevi = NEVI_PROMPT_TEMPLATE.format(context=nevi_context_sample[:400] + "…")

    print("\n  ── Nevi Prompt sent to Claude claude-opus-4-5 ──")
    print(SUBDIV)
    print(filled_nevi)
    print(SUBDIV)


# ── 4. CONCORDANT / NON-CONCORDANT CASE EXAMPLES ───────────────────────────
#
# REMOVED 2026-09-18. This section emitted hardcoded, fabricated case
# output: invented cases SCC-001/002 and NEVI-001/002 with made-up model
# responses, no retrieval distances, no chunk IDs, no source or page, and
# a miscitation of MPATH-Dx. It contradicted the real Section 2 output
# while being formatted to look exactly like it, which is precisely the
# kind of material that ends up in a supplement by accident.
#
# Nothing here was salvageable, because the problem was not formatting:
# there were no real cases behind it. The replacement below reads actual
# logs off disk and prints nothing at all when there are none.


def print_concordant_nonconcordant():
    """Print real concordant and non-concordant cases from the logs.

    Reads analysis_logs/ and results/, never invents a case, and prints a
    notice rather than an example when no run has happened yet.
    """
    print(box("SECTION 4 - CONCORDANT & NON-CONCORDANT CASES (FROM LOGS)"))

    results_dir = pathlib.Path("results")
    shown = 0

    for pathway, filename, ref_col, model_col in (
        ("CSCC", "cscc_cases.csv", "reference_grade", "model_grade"),
        ("Nevus", "nevus_cases.csv", "reference_stratum", "model_stratum"),
    ):
        table = results_dir / filename
        if not table.exists():
            continue

        with table.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        concordant = next((r for r in rows if r.get("concordant") == "1"), None)
        discordant = next((r for r in rows if r.get("concordant") == "0"), None)

        for label, row in (("CONCORDANT", concordant),
                           ("NON-CONCORDANT", discordant)):
            if not row:
                continue
            shown += 1
            print(SUBDIV)
            print(f"{pathway} - {label}")
            print(f"  case_id    : {row['case_id']}  rep{row.get('replicate')}")
            print(f"  reference  : {row.get(ref_col)}")
            print(f"  model      : {row.get(model_col)}")
            print(f"  confidence : {row.get('confidence_level')}")
            print(f"  images     : {row.get('magnifications_sent')}")
            if row.get("model_mpath_class"):
                print(f"  MPATH-Dx v2.0 : model={row['model_mpath_class']} "
                      f"expected={{{row.get('expected_mpath_classes')}}}"
                      f"{' (ambiguous under v2.0)' if row.get('mpath_ambiguous') == '1' else ''}")
            if row.get("consistency_flags"):
                print(f"  flags      : {row['consistency_flags']}")

            log_path = (pathlib.Path("analysis_logs") / pathway /
                        f"{row['case_id']}__rep{row.get('replicate')}.json")
            if log_path.exists():
                log = json.loads(log_path.read_text())
                print(f"  log        : {log_path}")
                print(f"  model_id   : {log['request'].get('model')}")
                print(f"  parse      : {log['parsing'].get('strategy_used')}")
            print(SUBDIV)

    if not shown:
        print("\n  No scored cases on disk yet.\n"
              "  Run:  python run_tests.py  then  python join_and_score.py\n"
              "  This section prints real cases only. It will stay empty\n"
              "  rather than print an illustration, because the previous\n"
              "  version's fabricated examples were indistinguishable from\n"
              "  real output once pasted into a document.\n")


HOW_TO = """
  1. python make_manifest.py --yes        freeze the run configuration
  2. python run_tests.py --dry-run        validate every case, spend nothing
  3. python run_tests.py                  the batch (resumable)
  4. python verify_logging.py             audit before scoring
  5. python join_and_score.py             concordance -> results/
  6. python generate_examples.py          this report, now from real logs
"""


def print_how_to():
    print(box("SECTION 5 - HOW TO CAPTURE LIVE EXAMPLES FROM YOUR OWN RUNS"))
    print(HOW_TO)


# ── 6. OPTIONAL LIVE LLM CALL ───────────────────────────────────────────────

def run_live_llm_example():
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n  ⚠  ANTHROPIC_API_KEY not set – skipping live LLM call.\n")
        return

    try:
        import anthropic
    except ImportError:
        print("  ⚠  anthropic package not installed – skipping live call.\n")
        return

    print(box("SECTION 6 — LIVE LLM CALL (text-only probe, no image)"))
    print("\n  Querying ChromaDB for real context, then prompting Claude…\n")

    client = chromadb.PersistentClient(path="./chroma_db")
    col    = client.get_collection("scc_grading_literature")
    r      = col.query(
        query_texts=["keratin pearls well differentiated squamous cell carcinoma"],
        n_results=3
    )
    context = "\n\n".join(r["documents"][0])

    probe_prompt = (
        "You are a dermatopathologist. Based only on this literature context:\n\n"
        f"{context[:1200]}\n\n"
        "Summarise in ≤4 bullet points the KEY histological features that distinguish "
        "Well Differentiated from Poorly Differentiated cutaneous SCC. "
        "Be concise and cite any grading system mentioned."
    )

    anthropic_client = anthropic.Anthropic(api_key=api_key)
    response = anthropic_client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=400,
        temperature=0.1,
        messages=[{"role": "user", "content": probe_prompt}],
    )

    print("  Context sent (first 400 chars of each chunk):")
    for i, doc in enumerate(r["documents"][0], 1):
        src = r["metadatas"][0][i-1].get("source", "?").replace("attached_assets/", "")
        print(f"\n    Chunk {i} [{src}]:")
        print(wrap(doc[:350] + "…"))

    print("\n  Claude response:")
    print(SUBDIV)
    for line in response.content[0].text.splitlines():
        print(f"  {line}")
    print(SUBDIV)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RAG+LLM examples report")
    parser.add_argument("--live", action="store_true",
                        help="Fire a real Claude API call (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    print("\n🔬  PATHOLOGY RAG SYSTEM — EXAMPLE GENERATOR")
    print("    SCC grading (72 chunks) + Nevi grading (294 chunks)\n")

    print_query_patterns()
    scc_data, nevi_data = print_retrieved_snippets()
    print_prompt_structure(scc_data, nevi_data)
    print_concordant_nonconcordant()
    print_how_to()

    if args.live:
        run_live_llm_example()

    print(f"\n{DIVIDER}")
    print("  Report complete. Use --live to also run a real LLM call.")
    print(f"{DIVIDER}\n")


if __name__ == "__main__":
    main()
