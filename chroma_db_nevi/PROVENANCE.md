# chroma_db_nevi provenance

**Rebuild, with one source substituted. Read before citing any nevus
retrieval number.**

Rebuilt 2026-09-18 by `build_vector_stores.py`. Three of the four source
documents are the original PDFs, byte-identical to the pre-migration
`run_manifest.json`. The fourth was lost and has been replaced by the
article text.

| Source | Status | Chunks (original → rebuild) |
|---|---|---|
| Nevi 2001_1749902667336.pdf | byte-identical PDF | 57 → 57 |
| Nevi 2025_1749902667337 | **text substitute** | 96 → 53 |
| Nevi MPath 2_1749902667338.pdf | byte-identical PDF | 95 → 95 |
| Nevi MPath_1749902667339.pdf | byte-identical PDF | 46 → 46 |
| **Total** | | **294 → 251** |

The substituted document is:

> Menzinger S, Merat R, Kaya G. Dysplastic Nevi and Superficial
> Borderline Atypical Melanocytic Lesions: Description of an Algorithmic
> Clinico-Pathological Classification. *Dermatopathology.* 2025;12(1):3.
> doi:10.3390/dermatopathology12010003

supplied as `attached_assets/Nevi 2025_1749902667337.txt`.

## What this means

The text carries the article's body, tables and figure legends, but not
the PDF's running headers, page furniture or reference list, so it
chunks to 53 rather than 96. Because chunk IDs are assigned sequentially
across the four documents in order, **every chunk ID after the
substituted document shifts**, and the nevus retrieval distances do not
match the pre-migration manifest. The original nevus reference value of
0.4066 for subquery 1 does not reproduce and cannot be made to.

This is recoverable only by restoring the original PDF (SHA-256
`f5c217b3e5bf2c2f14667d5606bd2a2c12eab30a0c48c2ea327ebc45040eef01`,
20 pages). If it turns up, drop it into `attached_assets/`, delete the
`.txt`, and run `python build_vector_stores.py --pathway Nevus --force`.
`build_vector_stores.py --check` reports current status.

## What is unaffected

- The store works. Retrieval returns sensible melanocytic literature.
- Grading does not depend on it. The MPATH-Dx v2.0 class definitions and
  the grading criteria live in the static prompt scaffold, in
  `mpath_dx.py` and the analyzer, not in retrieved text. Retrieval
  supplements the prompt; it is not the source of the decision rules.
- `chroma_db/` (CSCC) is unaffected and reproduces exactly.

## Reporting

Any figure, table or supplement quoting nevus retrieval distances must
either be regenerated against this store and labelled as post-migration,
or state that it came from the pre-migration store. Do not mix the two
in one table.
