# chroma_db provenance

**This is a rebuild, not the original store.**

The original `chroma_db/` did not survive the migration off Replit. This
directory was rebuilt on 2026-09-18 by `build_vector_stores.py` from the
two source PDFs in `attached_assets/`, both of which are byte-identical
to the ones recorded in the pre-migration `run_manifest.json`.

## Reproduction against the original

| | Original | Rebuild |
|---|---|---|
| chunk_count | 72 | 72 |
| chunk_id_fingerprint_sha256 | `8efff73f…dab8aa` | identical |
| corpus_fingerprint_sha256 | `99a4d98c…1146fd` | **differs** |
| subquery-1 top-1 | doc_64, 0.5462 | doc_64, 0.5462 |
| all 7 subquery top-1 distances | — | **identical to 4 dp** |

The corpus fingerprint differs because a newer `pypdf` extracts text
slightly differently: 7 of the 17 chunks that appear in the original
retrieval log differ, all by under 0.5% of their characters (whitespace
and ligature handling). None of that moved a retrieval result.

**Conclusion: published CSCC retrieval distances reproduce exactly
against this store.** Cite them without qualification. If a future
rebuild changes the source PDFs or the chunking parameters, re-check
before reusing this statement.

`chroma_db_nevi/` is a different situation: it cannot be rebuilt until
`Nevi 2025_1749902667337.pdf` is restored. Run
`python build_vector_stores.py --check`.
