"""
make_manifest.py
----------------
Step 1 of the logging workflow.  Run ONCE before any cases.

    python make_manifest.py

Writes run_manifest.json in the working directory.

Verifies:
  CSCC  chunk_count == 72,  subquery-1 top-1 distance == 0.5462
  Nevus chunk_count == 294, subquery-1 top-1 distance == 0.4066

If either check fails the script exits non-zero and must NOT be overridden
by re-embedding the store.  Published distances would become invalid.
"""

import hashlib
import json
import os
import pathlib
import platform
import sys
import importlib
import importlib.metadata
import datetime

import chromadb

# ── helpers ──────────────────────────────────────────────────────────────────

def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "?"


def session_id() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ") + "-grading"


# ── vector store fingerprint ──────────────────────────────────────────────────

def fingerprint_collection(col) -> dict:
    """Return chunk_id_fingerprint_sha256 and corpus_fingerprint_sha256."""
    # Retrieve ALL chunks (no filter, large limit)
    result = col.get(include=["documents", "metadatas"])
    ids   = result["ids"]
    docs  = result["documents"]

    # chunk_id fingerprint: sorted IDs joined by newline
    sorted_ids = sorted(ids)
    id_block   = "\n".join(sorted_ids)
    id_fp      = sha256_str(id_block)

    # corpus fingerprint: sorted (chunk_id + "\n" + text) blocks joined by newline
    pairs = sorted(zip(ids, docs), key=lambda x: x[0])
    corpus_block = "\n".join(f"{cid}\n{txt}" for cid, txt in pairs)
    corpus_fp    = sha256_str(corpus_block)

    return {"chunk_id_fingerprint_sha256": id_fp,
            "corpus_fingerprint_sha256":   corpus_fp}


def embedding_function_info(col) -> dict:
    """Read embedding function metadata at runtime from the collection object."""
    ef = col._embedding_function
    ef_class = type(ef).__module__ + "." + type(ef).__name__

    # Resolve model name: try common attribute names
    resolved_model = (
        getattr(ef, "model_name", None)
        or getattr(ef, "_model_name", None)
        or getattr(ef, "MODEL_NAME", None)
        or "all-MiniLM-L6-v2"   # known default for DefaultEmbeddingFunction
    )

    # Resolve dimensions by embedding a probe string
    try:
        probe = ef(["probe"])[0]
        dimensions = len(probe)
    except Exception:
        dimensions = 384  # known default for all-MiniLM-L6-v2

    return {"class": ef_class, "resolved_model": resolved_model,
            "dimensions": dimensions}


# ── source document metadata ──────────────────────────────────────────────────

def source_doc_meta(col, pdf_files: list[str]) -> list[dict]:
    """
    For each PDF, compute SHA-256, page count, and chunk count from the DB.
    """
    # Build source → chunk count map from DB metadata
    result   = col.get(include=["metadatas"])
    src_counts: dict[str, int] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "")
        src_counts[src] = src_counts.get(src, 0) + 1

    docs = []
    for pdf_path in pdf_files:
        filename = pathlib.Path(pdf_path).name
        if pathlib.Path(pdf_path).exists():
            file_sha = sha256_file(pdf_path)
            # Count pages with PyPDF
            try:
                from pypdf import PdfReader
                reader     = PdfReader(pdf_path)
                page_count = len(reader.pages)
            except Exception:
                page_count = 0
        else:
            file_sha   = ""
            page_count = 0

        # Match src_counts by any key that ends with the filename
        chunk_count = 0
        for src, cnt in src_counts.items():
            if pathlib.Path(src).name == filename or src == pdf_path:
                chunk_count += cnt

        docs.append({
            "filename":    filename,
            "sha256":      file_sha,
            "page_count":  page_count,
            "chunk_count": chunk_count,
        })
    return docs


# ── retrieval run ─────────────────────────────────────────────────────────────

def run_retrieval(col, subqueries: list[str], n_results: int = 5) -> dict:
    """Run all subqueries and return the full structured retrieval block."""
    all_results   = []
    all_chunk_ids = []   # in order, with repeats
    entry_n       = 1

    for q_idx, query in enumerate(subqueries, start=1):
        r = col.query(query_texts=[query], n_results=n_results)
        for rank, (doc, meta, dist, cid) in enumerate(zip(
                r["documents"][0],
                r["metadatas"][0],
                r["distances"][0],
                r["ids"][0]), start=1):
            all_results.append({
                "subquery_n": q_idx,
                "subquery":   query,
                "rank":       rank,
                "chunk_id":   cid,
                "distance":   round(dist, 4),
                "source":     pathlib.Path(meta.get("source", "")).name,
                "page":       meta.get("page", 0),
                "text":       doc,    # full, untruncated
            })
            all_chunk_ids.append(cid)
            entry_n += 1

    unique_ids      = list(dict.fromkeys(all_chunk_ids))   # ordered, de-duped
    unique_count    = len(unique_ids)
    duplicate_count = len(all_chunk_ids) - unique_count

    # Build context block exactly as the RAG system does (concatenate in order,
    # per-subquery top-n, joined by "\n\n")
    context_parts = []
    entry_i       = 0
    for _ in subqueries:
        for _ in range(n_results):
            if entry_i < len(all_results):
                context_parts.append(all_results[entry_i]["text"])
                entry_i += 1

    # Join with "\n\n" — no trailing separator.
    # Matches rag_system.py / nevi_rag_system.py get_grading_criteria()
    # and the verifier's CR13/NR13 rebuild formula.
    context_block = "\n\n".join(context_parts)
    context_sha   = sha256_str(context_block)

    return {
        "n_results_per_subquery":   n_results,
        "subqueries":               subqueries,
        "results":                  all_results,
        "unique_chunk_ids":         unique_ids,
        "unique_chunk_count":       unique_count,
        "duplicate_retrieval_count": duplicate_count,
        "context_block_chars":      len(context_block),
        "context_block_sha256":     context_sha,
    }


# ── prompt template extraction ────────────────────────────────────────────────

def extract_templates(analyzer_module_path: str) -> tuple[str, str]:
    """
    Read the create_analysis_prompt() body from source to capture the
    template text and SHA-256 before context substitution.
    Returns (system_template_text, user_template_text).
    The Anthropic calls in this codebase have no separate system parameter,
    so system_template_text is always "".
    """
    src = pathlib.Path(analyzer_module_path).read_text(encoding="utf-8")
    # Extract the string literal inside create_analysis_prompt
    # We look for the triple-quoted prompt = f"""...""" block
    import re
    # Match the f-string assigned to prompt inside create_analysis_prompt
    m = re.search(
        r'def create_analysis_prompt.*?prompt\s*=\s*f?"""(.*?)"""',
        src, re.DOTALL
    )
    if m:
        user_template = m.group(1)
    else:
        user_template = "(extraction failed – read source manually)"

    return "", user_template   # system is always "" in this codebase


# ── verification ──────────────────────────────────────────────────────────────

EXPECTED = {
    "CSCC":  {"chunk_count": 72,  "subq1_top1_distance": 0.5462},
    "Nevus": {"chunk_count": 294, "subq1_top1_distance": 0.4066},
}


def verify_pathway(pathway: str, retrieval: dict, col) -> list[str]:
    errors = []
    expected = EXPECTED[pathway]

    actual_count = col.count()
    if actual_count != expected["chunk_count"]:
        errors.append(
            f"{pathway}: chunk_count is {actual_count}, "
            f"expected {expected['chunk_count']}. "
            "Vector store has changed — published distances must be regenerated."
        )

    # subquery 1, rank 1 distance
    r1 = next((r for r in retrieval["results"]
               if r["subquery_n"] == 1 and r["rank"] == 1), None)
    if r1:
        actual_dist = r1["distance"]
        exp_dist    = expected["subq1_top1_distance"]
        if abs(actual_dist - exp_dist) > 0.0005:
            errors.append(
                f"{pathway}: subquery-1 top-1 distance is {actual_dist}, "
                f"expected {exp_dist}. "
                "Vector store has changed — published distances must be regenerated."
            )
    else:
        errors.append(f"{pathway}: could not find subquery-1 rank-1 result.")

    return errors


# ── CSCC pathway config ────────────────────────────────────────────────────────

CSCC_PDF_FILES = [
    "attached_assets/Grading differentiation.pdf",
    "attached_assets/Grading SCC diff.pdf",
]

CSCC_SUBQUERIES = [
    "well differentiated squamous cell carcinoma characteristics keratinization",
    "moderately differentiated squamous cell carcinoma features",
    "poorly differentiated squamous cell carcinoma criteria atypia",
    "keratin pearls horn cysts squamous cell carcinoma grading",
    "cellular atypia pleomorphism squamous cell carcinoma differentiation",
    "Broders grading system squamous cell carcinoma",
    "WHO grading squamous cell carcinoma differentiation",
]

CSCC_OUTPUT_SCHEMA = [
    "primary_grade", "confidence_level", "keratinization_present",
    "atypia_level", "key_features", "additional_observations",
]

# ── Nevus pathway config ──────────────────────────────────────────────────────

NEVUS_PDF_FILES = [
    "attached_assets/Nevi 2001_1749902667336.pdf",
    "attached_assets/Nevi 2025_1749902667337.pdf",
    "attached_assets/Nevi MPath 2_1749902667338.pdf",
    "attached_assets/Nevi MPath_1749902667339.pdf",
]

NEVUS_SUBQUERIES = [
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

NEVUS_OUTPUT_SCHEMA = [
    "traditional_grade", "mpath_grade", "confidence_level",
    "nuclear_abnormality_count", "architectural_features",
    "cytological_features", "grading_rationale", "clinical_significance",
]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    manifest_path = pathlib.Path("run_manifest.json")
    if manifest_path.exists():
        print("run_manifest.json already exists.")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    print("Opening ChromaDB collections…")
    cscc_client  = chromadb.PersistentClient(path="./chroma_db")
    nevus_client = chromadb.PersistentClient(path="./chroma_db_nevi")

    cscc_col  = cscc_client.get_collection("scc_grading_literature")
    nevus_col = nevus_client.get_collection("nevi_grading_literature")

    print(f"  CSCC  collection: {cscc_col.count()} chunks")
    print(f"  Nevus collection: {nevus_col.count()} chunks")

    # Fingerprints
    print("Fingerprinting CSCC store…")
    cscc_fp  = fingerprint_collection(cscc_col)
    print("Fingerprinting Nevus store…")
    nevus_fp = fingerprint_collection(nevus_col)

    # Embedding function info
    print("Reading embedding function metadata…")
    cscc_ef  = embedding_function_info(cscc_col)
    nevus_ef = embedding_function_info(nevus_col)

    # Source document metadata
    print("Computing source document SHA-256 and page counts…")
    cscc_docs  = source_doc_meta(cscc_col,  CSCC_PDF_FILES)
    nevus_docs = source_doc_meta(nevus_col, NEVUS_PDF_FILES)

    # Retrieval
    print("Running CSCC retrieval queries (7 × 5)…")
    cscc_retrieval  = run_retrieval(cscc_col,  CSCC_SUBQUERIES,  n_results=5)
    print("Running Nevus retrieval queries (12 × 5)…")
    nevus_retrieval = run_retrieval(nevus_col, NEVUS_SUBQUERIES, n_results=5)

    # Prompt templates
    print("Extracting prompt templates from source files…")
    cscc_sys,  cscc_user  = extract_templates("image_analyzer.py")
    nevus_sys, nevus_user = extract_templates("nevi_analyzer.py")

    # Build manifest
    sid = session_id()

    manifest = {
        "manifest_version": "1.0",
        "session_id":       sid,
        "created_utc":      datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),

        "environment": {
            "python_version": sys.version,
            "platform":       platform.platform(),
            "packages": {
                "chromadb":                pkg_version("chromadb"),
                "sentence-transformers":   pkg_version("sentence-transformers"),
                "langchain":               pkg_version("langchain"),
                "langchain-text-splitters":pkg_version("langchain-text-splitters"),
                "pypdf":                   pkg_version("pypdf"),
                "anthropic":               pkg_version("anthropic"),
                "pillow":                  pkg_version("pillow"),
            },
        },

        "pathways": {
            "CSCC": {
                "vector_store": {
                    "path":                     "chroma_db",
                    "collection":               "scc_grading_literature",
                    "chunk_count":              cscc_col.count(),
                    "chunk_id_fingerprint_sha256": cscc_fp["chunk_id_fingerprint_sha256"],
                    "corpus_fingerprint_sha256":   cscc_fp["corpus_fingerprint_sha256"],
                    "embedding_function":        cscc_ef,
                    "chunking": {
                        "splitter":      "RecursiveCharacterTextSplitter",
                        "chunk_size":    1000,
                        "chunk_overlap": 200,
                        "separators":    ["\n\n", "\n", " ", ""],
                    },
                    "source_documents": cscc_docs,
                },
                "retrieval": cscc_retrieval,
                "generation": {
                    "model_requested":      "claude-opus-4-5-20251101",
                    "temperature":          0.1,
                    "max_tokens":           1500,
                    "system_template_text": cscc_sys,
                    "system_template_sha256": sha256_str(cscc_sys),   # always compute, even for ""
                    "user_template_text":   cscc_user,
                    "user_template_sha256": sha256_str(cscc_user),
                },
                "output_schema_fields": CSCC_OUTPUT_SCHEMA,
            },

            "Nevus": {
                "vector_store": {
                    "path":                     "chroma_db_nevi",
                    "collection":               "nevi_grading_literature",
                    "chunk_count":              nevus_col.count(),
                    "chunk_id_fingerprint_sha256": nevus_fp["chunk_id_fingerprint_sha256"],
                    "corpus_fingerprint_sha256":   nevus_fp["corpus_fingerprint_sha256"],
                    "embedding_function":        nevus_ef,
                    "chunking": {
                        "splitter":      "RecursiveCharacterTextSplitter",
                        "chunk_size":    1000,
                        "chunk_overlap": 200,
                        "separators":    ["\n\n", "\n", " ", ""],
                    },
                    "source_documents": nevus_docs,
                },
                "retrieval": nevus_retrieval,
                "generation": {
                    "model_requested":      "claude-opus-4-5-20251101",
                    "temperature":          0.1,
                    "max_tokens":           1500,
                    "system_template_text": nevus_sys,
                    "system_template_sha256": sha256_str(nevus_sys),   # always compute, even for ""
                    "user_template_text":   nevus_user,
                    "user_template_sha256": sha256_str(nevus_user),
                },
                "output_schema_fields": NEVUS_OUTPUT_SCHEMA,
            },
        },
    }

    # Verification
    print("\nVerifying against published distances…")
    all_errors = []
    all_errors += verify_pathway("CSCC",  cscc_retrieval,  cscc_col)
    all_errors += verify_pathway("Nevus", nevus_retrieval, nevus_col)

    if all_errors:
        print("\n❌  VERIFICATION FAILED:")
        for e in all_errors:
            print(f"   • {e}")
        print("\nManifest NOT written. Resolve the above before proceeding.")
        sys.exit(1)

    print("✅  Verification passed.")

    # Write
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n✅  run_manifest.json written  (session_id: {sid})")
    print(f"   CSCC  context block: {cscc_retrieval['context_block_chars']:,} chars  "
          f"sha256 …{cscc_retrieval['context_block_sha256'][-12:]}")
    print(f"   Nevus context block: {nevus_retrieval['context_block_chars']:,} chars  "
          f"sha256 …{nevus_retrieval['context_block_sha256'][-12:]}")
    print(f"   CSCC  unique chunks: {cscc_retrieval['unique_chunk_count']}  "
          f"({cscc_retrieval['duplicate_retrieval_count']} duplicates)")
    print(f"   Nevus unique chunks: {nevus_retrieval['unique_chunk_count']}  "
          f"({nevus_retrieval['duplicate_retrieval_count']} duplicates)")
    print("\nNext: fill ground_truth_cscc.csv and ground_truth_nevus.csv, "
          "then run:  python run_tests.py")


if __name__ == "__main__":
    main()
