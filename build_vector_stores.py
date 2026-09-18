"""Rebuild the ChromaDB vector stores from the source PDFs.

    python build_vector_stores.py --pathway both [--force]
    python build_vector_stores.py --check        # verify sources only

Read this first
---------------
The original `chroma_db/` and `chroma_db_nevi/` directories did not
survive the migration off Replit. They were the reproducibility artefacts
behind the published retrieval distances (CSCC subquery-1 top-1 = 0.5462,
nevus = 0.4066), and they cannot be recovered - only rebuilt.

A rebuilt store will NOT reproduce those distances exactly. Chunk text is
deterministic given identical PDFs and splitter settings, but the
embedding values depend on the sentence-transformers and chromadb
versions in the environment, and those have moved since. Expect the
chunk IDs and counts to match and the distances to shift.

That is recoverable, but it has to be stated rather than discovered later:
any figure or supplement quoting the old distances must either be
regenerated against the new store or explicitly labelled as coming from
the pre-migration one. `--check` prints the old and new fingerprints side
by side so the difference is on the record.

This script refuses to overwrite an existing store without --force, so a
casual run cannot destroy a store that is currently backing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import config


SOURCES = {
    "CSCC": ["attached_assets/Grading differentiation.pdf",
             "attached_assets/Grading SCC diff.pdf"],
    "Nevus": ["attached_assets/Nevi 2001_1749902667336.pdf",
              "attached_assets/Nevi 2025_1749902667337.pdf",
              "attached_assets/Nevi MPath 2_1749902667338.pdf",
              "attached_assets/Nevi MPath_1749902667339.pdf"],
}

ID_PREFIX = {"CSCC": "doc", "Nevus": "nevi_doc"}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(manifest_path: str = config.RUN_MANIFEST) -> int:
    """Compare on-disk PDFs against the pre-migration manifest."""
    manifest = {}
    p = pathlib.Path(manifest_path)
    if p.exists():
        try:
            manifest = json.loads(p.read_text())
        except json.JSONDecodeError:
            pass

    missing = 0
    for pathway, files in SOURCES.items():
        recorded = {}
        if manifest:
            for doc in (manifest.get("pathways", {}).get(pathway, {})
                        .get("vector_store", {}).get("source_documents", [])):
                recorded[doc["filename"]] = doc

        print(f"\n{pathway}:")
        for rel in files:
            path = pathlib.Path(rel)
            name = path.name
            expected = recorded.get(name, {}).get("sha256")
            if not path.exists():
                missing += 1
                chunks = recorded.get(name, {}).get("chunk_count", "?")
                print(f"  MISSING          {name}  "
                      f"({chunks} chunks in the original store)")
                continue
            actual = sha256_file(path)
            if expected is None:
                print(f"  present          {name}  (not in manifest)")
            elif actual == expected:
                print(f"  byte-identical   {name}")
            else:
                missing += 1
                print(f"  DIFFERENT BYTES  {name}")
                print(f"      manifest {expected}")
                print(f"      on disk  {actual}")
    return missing


def build(pathway: str, force: bool) -> None:
    try:
        import chromadb
        from langchain_community.document_loaders import PyPDFLoader
        try:
            from langchain_text_splitters import (
                RecursiveCharacterTextSplitter)
        except ImportError:
            from langchain.text_splitter import (
                RecursiveCharacterTextSplitter)
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc}\n"
                 f"  pip install chromadb langchain-community pypdf")

    store_dir = pathlib.Path(config.CHROMA_DIR[pathway])
    if store_dir.exists():
        if not force:
            sys.exit(
                f"{store_dir} already exists. Rebuilding changes every "
                f"retrieval distance downstream, so pass --force only when "
                f"you intend that, and note the before/after values.")
        shutil.rmtree(store_dir)

    missing = [f for f in SOURCES[pathway]
               if not pathlib.Path(f).exists()
               and not pathlib.Path(f).with_suffix(".txt").exists()]
    if missing:
        sys.exit(f"cannot build {pathway}: missing source(s), with no .txt "
                 f"substitute either:\n  " + "\n  ".join(missing))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    texts, metadatas, ids = [], [], []
    doc_id = 0
    per_file = {}
    substituted = []
    for rel in SOURCES[pathway]:
        path = pathlib.Path(rel)
        if path.exists():
            docs = PyPDFLoader(rel).load()
        else:
            # Text substitute: the original PDF was lost and the article
            # text was supplied instead. Chunk boundaries will NOT match
            # the PDF's, so retrieval distances shift. Recorded in the
            # metadata and reported at the end so it cannot pass silently.
            text_path = path.with_suffix(".txt")
            from langchain_core.documents import Document
            docs = [Document(page_content=text_path.read_text(encoding="utf-8"),
                             metadata={"page": 0})]
            substituted.append(path.name)
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            texts.append(chunk.page_content)
            metadatas.append({"source": rel,
                              "page": chunk.metadata.get("page", 0),
                              "source_substituted": path.name in substituted})
            ids.append(f"{ID_PREFIX[pathway]}_{doc_id}")
            doc_id += 1
        per_file[pathlib.Path(rel).name] = len(chunks)

    client = chromadb.PersistentClient(path=str(store_dir))
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION[pathway])
    collection.add(documents=texts, metadatas=metadatas, ids=ids)

    # Same definition make_manifest.fingerprint_collection uses (sorted
    # IDs joined by newline), so the two are directly comparable.
    fingerprint = hashlib.sha256(
        "\n".join(sorted(ids)).encode()).hexdigest()
    print(f"\n{pathway}: {len(texts)} chunks into {store_dir}")
    for name, count in per_file.items():
        print(f"  {count:4} {name}")
    print(f"  chunk_id_fingerprint_sha256 = {fingerprint}")
    if substituted:
        print("\n  TEXT SUBSTITUTE USED for: " + ", ".join(substituted))
        print("  The original PDF was lost and the article text was supplied "
              "instead.\n  Chunk boundaries differ from the PDF's, so every "
              "chunk id after the\n  substituted document shifts and the "
              "retrieval distances for this\n  pathway will NOT match the "
              "pre-migration manifest. Say so anywhere\n  the nevus numbers "
              "are reported.")
    print("\n  Re-run make_manifest.py and record the new values.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pathway", choices=["CSCC", "Nevus", "both"],
                    default="both")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        problems = check_sources()
        print()
        if problems:
            print(f"{problems} source file(s) missing or changed. The store "
                  f"cannot be rebuilt to match the original until they are "
                  f"restored.")
            sys.exit(1)
        print("All source PDFs present and byte-identical to the manifest.")
        return

    for pathway in (["CSCC", "Nevus"] if args.pathway == "both"
                    else [args.pathway]):
        build(pathway, args.force)


if __name__ == "__main__":
    main()
