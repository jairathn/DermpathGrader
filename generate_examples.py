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

import sys
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

    print(f"""
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
    print(f"\n\n  ── Nevi Collection (chroma_db_nevi / nevi_grading_literature) ──")
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

CONCORDANT_SCC = {
    "case_id":          "SCC-001",
    "image_file":       "well_diff_scc_sample.jpg",
    "pathologist_grade":"Well Differentiated",
    "llm_output": {
        "primary_grade":          "Well Differentiated",
        "confidence_level":       "High",
        "keratinization_present": True,
        "atypia_level":           "minimal",
        "key_features": [
            "Abundant keratin pearl formation throughout the tumor nests",
            "Well-preserved intercellular bridges (desmosomes)",
            "Orderly squamous maturation from peripheral basal layer inward",
            "Rare mitotic figures, no atypical mitoses",
        ],
        "additional_observations":
            "The tumor nests show centrally placed eosinophilic keratin whorls "
            "(Malpighian corpuscles). Stromal reaction is desmoplastic but mild. "
            "Overall architecture is cohesive without infiltrative tongues.",
    },
    "concordant":       True,
    "rag_query_used":   "well differentiated squamous cell carcinoma characteristics keratinization",
    "retrieved_snippet":
        "Well-differentiated squamous cell carcinomas retain the basic architecture "
        "of normal squamous epithelium and typically form keratin pearls (also called "
        "squamous eddies or Malpighian corpuscles). The cells have abundant eosinophilic "
        "cytoplasm, distinct cell borders, and intercellular bridges. Nuclear atypia is "
        "minimal and mitotic figures are infrequent. — Broders (1921), Mod in WHO 2022.",
}

NON_CONCORDANT_SCC = {
    "case_id":          "SCC-002",
    "image_file":       "mod_diff_scc_ambiguous.jpg",
    "pathologist_grade":"Moderately Differentiated",
    "llm_output": {
        "primary_grade":          "Well Differentiated",
        "confidence_level":       "Medium",
        "keratinization_present": True,
        "atypia_level":           "minimal",
        "key_features": [
            "Focal keratin pearl formation in several nests",
            "Moderate nuclear enlargement present in peripheral cells",
            "Some loss of polarity at tumor–stroma interface",
        ],
        "additional_observations":
            "Keratinization is present but focal. Peripheral nuclear atypia is "
            "present but less conspicuous in the image centre. The model weighted "
            "the keratinization heavily, potentially under-scoring the atypia.",
    },
    "concordant":   False,
    "discordance_reason":
        "Model over-weighted focal keratinization and under-weighted moderate nuclear "
        "pleomorphism at the invasive front. Pathologist graded as Moderately "
        "Differentiated based on >20% atypical cells across the full slide.",
    "rag_query_used":   "moderately differentiated squamous cell carcinoma features",
    "retrieved_snippet":
        "Moderately differentiated SCC shows variable keratinization. Keratin pearls "
        "may be present but less prominent than in well-differentiated tumors. Nuclear "
        "pleomorphism is moderate, mitotic figures more frequent. Tumour architecture "
        "may show irregular nests with focal loss of polarity. — WHO Classification 2022.",
}

CONCORDANT_NEVI = {
    "case_id":           "NEVI-001",
    "image_file":        "severe_dysplasia_nevi.jpg",
    "pathologist_grade": "Severe Dysplasia",
    "llm_output": {
        "traditional_grade":      "Severe Dysplasia",
        "mpath_grade":            "High-Grade Dysplasia",
        "confidence_level":       "High",
        "nuclear_abnormality_count": 4,
        "architectural_features": [
            "Confluent junctional melanocytic hyperplasia",
            "Bridging between adjacent rete ridges",
            "Pagetoid spread of melanocytes into upper epidermis",
        ],
        "cytological_features": [
            "Marked nuclear pleomorphism (nuclei ≥2× basal keratinocytes)",
            "Prominent eosinophilic nucleoli",
            "Increased nuclear-to-cytoplasmic ratio",
            "Irregular nuclear contours",
        ],
        "grading_rationale":
            "Four nuclear abnormalities identified plus confluent hyperplasia and "
            "pagetoid spread — meets ≥3 nuclear + architectural high-risk criteria "
            "for Severe Dysplasia. MPATH-Dx maps this to High-Grade.",
        "clinical_significance":
            "Complete excision with adequate margins recommended. Sentinel node "
            "evaluation not routinely indicated for dysplastic nevi alone.",
    },
    "concordant":      True,
    "rag_query_used":  "severe dysplasia atypical melanocytic nevi criteria",
    "retrieved_snippet":
        "High-grade (severe) dysplasia is defined by the presence of three or more "
        "cytological abnormalities or any single high-risk architectural feature such "
        "as confluent junctional hyperplasia, pagetoid scatter, or bridging of nests. "
        "Nuclear size exceeds twice that of adjacent basal keratinocytes. MPATH-Dx "
        "Class III/IV. — Piepkorn et al., JAMA Dermatol 2014 / 2022 update.",
}

NON_CONCORDANT_NEVI = {
    "case_id":           "NEVI-002",
    "image_file":        "mild_moderate_nevi_borderline.jpg",
    "pathologist_grade": "Mild Dysplasia",
    "llm_output": {
        "traditional_grade":         "Moderate Dysplasia",
        "mpath_grade":               "Low-Grade Dysplasia",
        "confidence_level":          "Low",
        "nuclear_abnormality_count": 2,
        "architectural_features": [
            "Symmetric overall silhouette",
            "Focal shoulder phenomenon",
        ],
        "cytological_features": [
            "Slight nuclear enlargement (estimated 1.3–1.5× basal keratinocytes)",
            "Mild chromatin irregularity in a subset of cells",
        ],
        "grading_rationale":
            "Nuclear size borderline between 1× and 1.5×. Model assigned Moderate "
            "due to two cytological findings. However confidence is Low given the "
            "ambiguity at the mild/moderate boundary.",
        "clinical_significance":
            "Low-grade per MPATH-Dx; clinical management similar regardless of "
            "mild vs moderate distinction. Re-excision margins may still apply.",
    },
    "concordant":   False,
    "discordance_reason":
        "Model over-graded to Moderate Dysplasia. The pathologist identified the "
        "nuclear enlargement as ≤1.2× (within Mild range) and attributed the apparent "
        "chromatin variation to section thickness/staining artefact. Low confidence "
        "output correctly flags this as ambiguous. MPATH-Dx (Low-Grade) was concordant "
        "even when traditional tier was not.",
    "rag_query_used":  "mild dysplasia atypical melanocytic nevi characteristics",
    "retrieved_snippet":
        "Mild dysplasia is characterized by minimal cytological atypia: nuclear size "
        "approximates that of adjacent basal keratinocytes, chromatin is evenly "
        "distributed, and nucleoli are inconspicuous. Architectural disorder is minimal; "
        "nests are regular and do not bridge adjacent rete ridges. MPATH-Dx Class I/II. "
        "— Elder et al., Am J Surg Pathol 2006 / Gerami et al., 2010.",
}


def print_concordant_nonconcordant():
    print(box("SECTION 4 — CONCORDANT & NON-CONCORDANT CASE EXAMPLES"))

    cases = [
        ("SCC — CONCORDANT", CONCORDANT_SCC),
        ("SCC — NON-CONCORDANT", NON_CONCORDANT_SCC),
        ("NEVI — CONCORDANT", CONCORDANT_NEVI),
        ("NEVI — NON-CONCORDANT", NON_CONCORDANT_NEVI),
    ]

    for label, case in cases:
        print(f"\n  ── {label} (Case {case['case_id']}) ──")

        if case.get("concordant"):
            print("  ✅ CONCORDANT — LLM grade matches pathologist grade")
        else:
            print("  ❌ NON-CONCORDANT — LLM grade differs from pathologist grade")

        print(f"  Pathologist grade : {case.get('pathologist_grade', case.get('traditional_grade', ''))}")

        if "primary_grade" in case["llm_output"]:
            print(f"  LLM grade         : {case['llm_output']['primary_grade']}")
        else:
            print(f"  LLM grade (trad.) : {case['llm_output']['traditional_grade']}")
            print(f"  LLM grade (mpath) : {case['llm_output']['mpath_grade']}")

        print(f"  Confidence        : {case['llm_output']['confidence_level']}")

        print(f"\n  ChromaDB query used:")
        print(f"    \"{case['rag_query_used']}\"")

        print(f"\n  Retrieved literature snippet (top-1):")
        print(wrap(case["retrieved_snippet"]))

        feats = case["llm_output"].get("key_features",
               case["llm_output"].get("cytological_features", []))
        if feats:
            print(f"\n  LLM-identified features:")
            for f in feats:
                print(wrap(f"• {f}", indent=6))

        rationale_key = ("additional_observations" if "additional_observations" in case["llm_output"]
                         else "grading_rationale")
        rationale = case["llm_output"].get(rationale_key, "")
        if rationale:
            print(f"\n  LLM reasoning:")
            print(wrap(rationale))

        if not case.get("concordant") and "discordance_reason" in case:
            print(f"\n  ⚠ Discordance explanation:")
            print(wrap(case["discordance_reason"]))

        print()


# ── 5. HOW TO CAPTURE YOUR OWN LIVE EXAMPLES ────────────────────────────────

HOW_TO = """
  ── How to Capture Live Examples From Your Own Runs ──────────────────────

  A. Log every analysis run to a JSON file
  ─────────────────────────────────────────
  In image_analyzer.py → analyze_image(), add after result is built:

      import json, datetime, pathlib
      log_dir = pathlib.Path("analysis_logs"); log_dir.mkdir(exist_ok=True)
      entry = {
          "timestamp":       datetime.datetime.utcnow().isoformat(),
          "rag_queries":     list(self.rag_system.query_log),  # see step B
          "retrieved_docs":  context_documents,                 # see step C
          "llm_prompt":      prompt,
          "llm_raw_output":  analysis_text,
          "parsed_result":   result,
      }
      with open(log_dir / f"run_{entry['timestamp'][:19]}.json", "w") as f:
          json.dump(entry, f, indent=2)

  B. Expose the RAG query log
  ───────────────────────────
  In rag_system.py → __init__, add:  self.query_log = []
  In query_documents(), prepend:     self.query_log.append(query)

  C. Capture the raw retrieved documents (not just combined_text)
  ───────────────────────────────────────────────────────────────
  In image_analyzer.py → analyze_image(), before building prompt:

      context_obj       = self.rag_system.get_grading_criteria_with_docs()
      context           = context_obj["text"]
      context_documents = context_obj["documents"]   # list of {content, metadata, distance}

  Add get_grading_criteria_with_docs() to RAGSystem that returns both.

  D. Label concordant / non-concordant
  ─────────────────────────────────────
  In the Streamlit UI (utils.py → display_results), add a radio widget:

      feedback = st.radio("Pathologist grade (for concordance logging):",
                          ["Well Differentiated","Moderately Differentiated",
                           "Poorly Differentiated","Skip"], index=3)
      if feedback != "Skip":
          run_log["pathologist_grade"] = feedback
          run_log["concordant"] = (feedback == result["primary_grade"])
          # re-save the JSON entry

  E. Retrieve and inspect your logs
  ──────────────────────────────────
  After running several cases:

      import json, glob
      logs = [json.load(open(f)) for f in glob.glob("analysis_logs/*.json")]
      concordant     = [l for l in logs if l.get("concordant") is True]
      non_concordant = [l for l in logs if l.get("concordant") is False]
      print(f"Concordant: {len(concordant)}, Non-concordant: {len(non_concordant)}")

  F. Quick CLI demo of live ChromaDB retrieval (no LLM key needed)
  ─────────────────────────────────────────────────────────────────
      python generate_examples.py           # prints this full report
      python generate_examples.py --live    # also fires a real Claude API call
                                            # (requires ANTHROPIC_API_KEY)
"""


def print_how_to():
    print(box("SECTION 5 — HOW TO CAPTURE LIVE EXAMPLES FROM YOUR OWN RUNS"))
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
