# Study design, protocol v2.2

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

## What the model reports beyond the label

Both pathways return, per case, the fields a sign-out would carry:
specimen adequacy (adequate / limited), a ranked
differential of two to four diagnoses, and the ancillary studies the
model would order at sign-out (PRAME, Melan-A/MART-1, SOX10, HMB-45,
Ki-67, p16, FISH, NGS, deeper levels for melanocytic; p63/p40, CK5/6,
S100/SOX10, Ber-EP4 and others for CSCC). CSCC cases additionally carry
histologic subtype, Broders grade, depth of invasion and the high-risk
features that move a tumour between stages (perineural and
lymphovascular invasion, invasion beyond fat, thickness over 6 mm,
desmoplastic subtype, bone invasion). Melanoma cases carry ulceration
and mitotic rate alongside Breslow.

None of these are scored as the primary outcome. They exist so that a
discordant case can be read, so that differential and ancillary
behaviour can be described as secondary outcomes, and so that
`report.py` can render a synoptic report a reader can compare against
their own. Every field is enforced by the JSON schema; a response that
does not match it is an error, not a partial result.

## Forced choice

Every case gets a grade. There is no nondiagnostic class, no
nondiagnostic lesion category, and no nondiagnostic adequacy value; the
word does not appear in either prompt, and the schema makes such a
response unemittable.

This is a design decision, not an oversight. An opt-out lets the model
decline precisely the cases it finds hardest, which does two things at
once: it inflates apparent accuracy on the cases it does answer, and it
makes the result incomparable with a reader who was obliged to commit.
Report a forced-choice number or report a decline rate, but a mixture of
the two is not interpretable.

The published MPATH-Dx v2.0 schema does define a Class 0 for
nondiagnostic material. `mpath_dx.CLASSES` retains it so the module
stays a faithful record; `mpath_dx.GRADEABLE_CLASSES` is the subset this
protocol permits.

Nothing is lost by this. `specimen_adequacy` still separates a clean
section from a limited one, `confidence_level` still records how sure
the model is, and the rationale still says what could not be assessed.
Those travel with a committed grade instead of replacing it, so
low-confidence and limited-specimen cases can be analysed as their own
stratum after the fact - which is more useful than a non-answer, because
the grade is still there to be checked.

If genuinely ungradeable material reaches the grader, that is a
case-selection problem: the 350 cases are curated and carry reference
diagnoses. Fix it in the registry rather than handing the grader a way
around it.

## Replicates and test-retest

Three successful replicates per case (minimum two). This model family
does not accept `temperature`, so run-to-run variation is whatever the
model produces at fixed effort; replicates measure it rather than
control it. The scorer reports, per pathway, the fraction of cases where
every replicate agreed and Fleiss' kappa across replicates, and uses the
per-case majority label for the reader comparison.

## Refusals, truncation and failures

A case the model declines to grade (`stop_reason == "refusal"`) is not a
grade. It is logged with the API's stated category, counted in
`parser_summary.csv`, excluded from concordance, and reported. The
batch runner retries the case up to two extra times; a persistent
refusal is left on the record rather than worked around. **No model
fallback is enabled**: substituting a different model for refused cases
would make the batch two experiments.

A response cut off at `max_tokens` is likewise an error, never parsed.
Transient API failures retry with backoff, and the attempt count and
each failed attempt's error are in the case log.

## Reader study

Readers grade from `.svs` in their own viewer, in a per-reader shuffled
order under opaque reading IDs (`build_case_registry.py
--reader-manifest`). Their returned grades go in
`data/reader_grades_<pathway>.csv` as `reader_id, case_id, grade`.

`join_and_score.py` then writes `reader_comparison.csv` with, on the
same case set: model vs reference, each reader vs reference, model vs
each reader, every reader pair, model vs reader consensus, and reader
consensus vs reference — each with agreement, a Wilson interval,
unweighted and quadratic-weighted kappa. The model's label in that table
is its majority across replicates. The headline comparison is the first
two rows against each other.

## Order of operations

```
doctor.py                          # environment, stores, key, registry
build_vector_stores.py --check     # sources intact?
build_vector_stores.py             # build the stores
build_case_registry.py --scan      # registry, check 50/stratum, Class II mix
extract_tiles.py --registry ...    # .svs -> four JPEGs
build_case_registry.py --reader-manifest   # blinded reader order
make_manifest.py --yes             # freeze the configuration
run_tests.py --dry-run             # validate + cost estimate, no API spend
run_tests.py --workers 4           # the batch; re-run to fill gaps
verify_logging.py                  # audit before scoring
join_and_score.py                  # concordance, replicates, readers
report.py --all                    # synoptic report per case
```

## Open decisions

- The Class II mix: how many high-grade dysplastic nevi versus melanoma
  in situ. This is the one composition choice the class label cannot
  record for you.
- How many readers, and does each read all 350 cases?
