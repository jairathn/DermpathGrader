# data/

| File | Written by | Read by |
|---|---|---|
| `case_registry.csv` | `build_case_registry.py --scan` (or by hand) | `run_tests.py`, `extract_tiles.py`, `doctor.py` |
| `tile_coords.csv` | a person, only for `extract_tiles.py --source curated` | `extract_tiles.py` |
| `reader_manifest.csv` | `build_case_registry.py --reader-manifest` | handed to readers |
| `reader_grades_nevus.csv`, `reader_grades_cscc.csv` | a person, from the readers' returned sheets | `join_and_score.py` |

Reader grade files have three columns: `reader_id, case_id, grade`.
Melanocytic grades are MPATH-Dx v2.0 classes (`I`, `II`, `III`, `IV`);
CSCC grades are `Well Differentiated` / `Moderately Differentiated` /
`Poorly Differentiated` (common spellings are normalised). A grade that
cannot be recognised is dropped and counted, never guessed at. The
`.example.csv` files show the shape; copy one and remove `.example`.

Ground truth lives at the repo root (`ground_truth_cscc.csv`,
`ground_truth_nevus.csv`) and is never passed to an analyzer.
