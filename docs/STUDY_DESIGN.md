# Study design, protocol v2.0

Dated 2026-09-18. This is the design the code implements. Where a design
decision has a consequence the code cannot fix, it is stated here rather
than left to be discovered during analysis.

## Cases

350 total, 50 per stratum.

| Pathway | Strata | n |
|---|---|---|
| Melanocytic | MPATH-Dx v2.0 Class I, II, III, IV | 200 |
| CSCC | well, moderately, poorly differentiated | 150 |

The melanocytic strata are the MPATH-Dx v2.0 classes themselves. Four
classes at 50 each is the same 200 cases the earlier three-tier-plus-
melanoma design called for, so the collection burden is unchanged.

## Two arms, two input formats

| | Dermatopathologist readers | Model |
|---|---|---|
| Input | whole-slide `.svs` | 4 JPEGs per case |
| Navigation | free pan and zoom | four fixed frames |
| Magnifications | any | whole slide, 4x, 10x, 40x |

The vision API cannot read `.svs`, which is why the arms differ. That
asymmetry is the main methodological feature of this design, and it cuts
in a specific direction: **the reader arm and the model arm are not
receiving the same information**, so a difference in accuracy between
them is not purely a difference in diagnostic ability.

Three things follow.

**1. Field selection.** `extract_tiles.py` defaults to `--source auto`: a
deterministic tissue-centroid rule, no human in the loop, reproducible
from the slide. `--source curated` reads coordinates from
`data/tile_coords.csv` where the automatic frame is useless. The method
used is recorded per case in `tile_selection_method`. One line in the
limitations paragraph covers it.

**2. Four frames is a protocol constant, not a per-case choice.** A case
graded on three images is not comparable to one graded on four, so a
missing magnification is a hard error everywhere in the pipeline.

**3. The comparison to state.** This design measures the model on
fixed-field JPEGs against readers on whole slides. It does not measure
"model versus pathologist" in the abstract. If the intended claim is the
latter, the reader arm would need to be restricted to the same four
frames, which is a different and considerably more expensive study.

## Grading outputs

### CSCC
`primary_grade` (well / moderately / poorly differentiated), plus
keratinization, atypia level, key features, and per-magnification
evidence.

### Melanocytic
Two independent judgements, plus supporting fields:

- `stratum_label` — mild / moderate / severe / **melanoma**. The
  four-way study label.
- `lesion_category`, `dysplasia_grade`, `melanoma_subtype`
  (in situ / invasive), `breslow_estimate_mm`.
- `mpath_dx_v2_class` — 0, I, II, III or IV.

Melanoma is a category, not the top of the dysplasia ladder. v1 could
only emit three dysplasia tiers, so a melanoma had nowhere to go.

## MPATH-Dx v2.0 and the moderate stratum

Barnhill RL, Elder DE, Piepkorn MW, et al. Revision of the Melanocytic
Pathology Assessment Tool and Hierarchy for Diagnosis Classification
Schema for Melanocytic Lesions: A Consensus Statement. *JAMA Netw Open.*
2023;6(1):e2250613. doi:10.1001/jamanetworkopen.2022.50613

| Class | Definition |
|---|---|
| 0 | Nondiagnostic |
| I | Low-grade atypia; nuclei <1.5x resting basal keratinocyte nuclei |
| II | High-grade atypia; nuclei ≥1.5x to >2x. **Includes melanoma in situ** |
| III | Invasive melanoma, Breslow <0.8 mm (pT1a) |
| IV | Invasive melanoma, Breslow ≥0.8 mm (≥pT1b) |

**Sampling the classes directly removes the problem the three-tier design
had.** v2.0 produced its four classes by deleting v1.0's standalone
moderate-atypia category, because observers could not reproduce that
split. Class I is defined as mild-to-moderate and Class II as high-end
moderate-to-severe, so a case labelled "moderate dysplasia" had no single
correct class. With classes as the strata, every case has exactly one
correct answer, assigned by the reference dermatopathologist.

Legacy three-tier labels are not converted automatically anywhere in the
pipeline. `mpath_dx.class_from_dysplasia_grade()` returns a set, and
"moderate" returns `{I, II}`; those cases need a dermatopathologist to
pick one before entering the study. The three-tier grade is still
recorded per case in `dysplasia_grade`, so mild/moderate/severe can be
cross-tabulated against class in the analysis.

### The one thing to compose deliberately: Class II

Class II contains **both** high-grade dysplastic nevi **and** melanoma in
situ. They are the same class under v2.0, and the class label alone does
not distinguish them. Three consequences:

- Decide the Class II mix in advance and record it. A Class II stratum
  that is 90% dysplasia measures something different from one that is
  90% in situ melanoma. `build_case_registry.py` prints the composition.
- Melanoma detection is scored on `lesion_category`, not on class,
  because a Class II answer is not a melanoma answer.
- `Nevus_insitu_vs_invasive` is a separate arm. In situ versus invasive
  is what moves a case from Class II to Class III or IV, and depth of
  invasion is the hardest thing to judge from four fixed frames, so it is
  where the model arm is most likely to fail.

### Melanoma reporting

Every melanoma case carries:

| Field | Purpose |
|---|---|
| `melanoma_subtype` | in situ or invasive - separates Class II from III/IV |
| `melanoma_histologic_subtype` | lentigo maligna, superficial spreading, nodular, acral lentiginous, desmoplastic, other |
| `breslow_estimate_mm` | separates Class III (<0.8 mm) from Class IV (≥0.8 mm) |
| `ulceration_present`, `mitoses_per_mm2` | recorded where assessable |

Breslow is scored two ways: agreement on the 0.8 mm cutoff (which is what
changes the class) and mean absolute error in millimetres.

## Scoring arms

CSCC: one arm, three-way exact match, with unweighted and quadratic-
weighted kappa.

Melanocytic: four, plus a Breslow arm.

1. `Nevus_MPATH_v2_class` — exact Class I/II/III/IV match. Primary.
2. `Nevus_management_binary` — Class I versus Class II+, the re-excision
   decision. The arm that corresponds to something happening to a patient.
3. `Nevus_melanoma_detection` — sensitivity and specificity for melanoma,
   scored on `lesion_category` rather than class, because Class II holds
   both high-grade dysplasia and melanoma in situ.
4. `Nevus_insitu_vs_invasive` — among reference melanomas, whether in
   situ and invasive are told apart.
5. `Nevus_breslow::pT1b_cutoff_agreement` — agreement on the 0.8 mm
   cutoff, with mean absolute error reported alongside.

`Nevus_internal_consistency` is reported alongside but is not a
concordance arm: it measures whether the model's own fields agree with
each other.

All proportions carry Wilson intervals, which behave at 50/50 where the
normal approximation does not.

## Replicates

Minimum two, default three. This model family does not accept
`temperature`, so run-to-run variation is whatever the model produces at
fixed effort. Replicates measure it; they do not control it.

## Order of operations

```
build_vector_stores.py --check     # sources intact?
build_vector_stores.py             # build the stores
build_case_registry.py --scan      # registry, check 50/stratum
extract_tiles.py --registry ...    # .svs -> four JPEGs
build_case_registry.py --reader-manifest   # blinded reader order
make_manifest.py                   # freeze the configuration
run_tests.py --dry-run             # validate, no API spend
run_tests.py --replicates 3        # the batch
verify_logging.py                  # audit before scoring
join_and_score.py                  # concordance
```

## Open decisions

- The Class II mix: how many high-grade dysplastic nevi versus melanoma
  in situ. This is the one composition choice the class label cannot
  record for you.
- How many readers, and does each read all 350 cases?
