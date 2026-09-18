# Study design, protocol v2.0

Dated 2026-09-18. This is the design the code implements. Where a design
decision has a consequence the code cannot fix, it is stated here rather
than left to be discovered during analysis.

## Cases

350 total, 50 per stratum.

| Pathway | Strata | n |
|---|---|---|
| Melanocytic | mild dysplasia, moderate dysplasia, severe dysplasia, melanoma | 200 |
| CSCC | well, moderately, poorly differentiated | 150 |

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

**1. Field selection is a hidden diagnostic step.** A reader decides
where to look at 40x *after* seeing the whole slide, and can go back. The
model cannot. Whoever chooses the 4x/10x/40x coordinates has therefore
already made part of the diagnosis, and if that person knows the
reference label, the label leaks into the model's input. `extract_tiles.py`
supports two methods and records which was used per case:

- `curated` — a person picks the fields. Record who, and whether they
  were blinded, in `tiles_chosen_by` and `tiles_chooser_blinded`. If the
  chooser was unblinded, the model arm's accuracy is an upper bound, not
  an estimate.
- `auto` — a deterministic tissue-centroid rule, no human. Nothing leaks.
  The frame may miss the diagnostic area, which is a real result about
  fixed-frame grading rather than a bug.

Do not mix methods within a stratum; that confounds selection method with
grade. The cleanest design is `auto` for everything, with `curated` as a
pre-registered secondary analysis.

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

**There is a direct tension between this schema and a 50-case moderate
stratum, and it should be stated in the manuscript rather than worked
around.** v2.0 produced four classes precisely by deleting v1.0's
standalone moderate-atypia class, because observers could not reproduce
that split. Class I is defined as mild-to-moderate and Class II as
high-end moderate-to-severe. A moderate case therefore has no single
correct v2.0 class.

The code handles this by scoring the moderate stratum against `{I, II}`
and reporting those cases separately, so the headline MPATH-Dx number is
never quietly inflated by a category that cannot be wrong. Two options if
a single class per case is wanted:

- Have the reference dermatopathologist assign a class per case into
  `mpath_dx_v2_reference`. The scorer uses it when present. This is the
  defensible route.
- Drop the moderate stratum and sample Class I and Class II directly,
  which aligns the design to the schema but abandons the three-tier
  comparison.

**Second tension: melanoma in situ is Class II**, alongside high-grade
dysplasia. If the melanoma stratum contains in situ cases, they are not
separable from severe dysplasia on the MPATH-Dx arm. Record
`melanoma_subtype` for every melanoma case. If the intended claim is
about melanoma detection, the melanoma-detection and four-way arms carry
it; the MPATH-Dx arm does not.

## Scoring arms

CSCC: one arm, three-way exact match, with unweighted and quadratic-
weighted kappa.

Melanocytic: four.

1. `Nevus_stratum_4way` — exact four-way match.
2. `Nevus_MPATH_v2_class` — model class within the expected set, reported
   overall and split into unambiguous and ambiguous (moderate) strata.
3. `Nevus_management_binary` — Class I versus Class II+, the re-excision
   decision. The arm that corresponds to something happening to a patient.
4. `Nevus_melanoma_detection` — sensitivity and specificity for melanoma.

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

- Does the melanoma stratum contain in situ cases, invasive, or both?
  The registry supports all three; the scoring consequences differ, per
  above.
- Is field selection `curated` or `auto` for the primary analysis?
- How many readers, and does each read all 350 cases?
