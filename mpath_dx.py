"""MPATH-Dx version 2.0 support (melanocytic pathway only).

Source
------
Barnhill RL, Elder DE, Piepkorn MW, Knezevich SR, Reisch LM, Eguchi MM,
Bastian BC, Blokx W, Bosenberg M, Busam KJ, Carr R, Cochran A, Cook MG,
Duncan LM, Elenitsas R, de la Fouchardiere A, Gerami P, Johansson I,
Ko J, Landman G, Lazar AJ, Lowe L, Massi D, Messina J, Mihic-Probst D,
Parker DC, Schmidt B, Shea CR, Scolyer RA, Tetzlaff M, Xu X, Yeh I,
Zembowicz A, Elmore JG. Revision of the Melanocytic Pathology Assessment
Tool and Hierarchy for Diagnosis Classification Schema for Melanocytic
Lesions: A Consensus Statement. JAMA Netw Open. 2023;6(1):e2250613.
doi:10.1001/jamanetworkopen.2022.50613

What changed from the previous implementation
---------------------------------------------
The v1 code did not implement MPATH-Dx v1.0's five classes at all. It
emitted a two-tier `mpath_grade` of "Low-Grade Dysplasia" /
"High-Grade Dysplasia", and the nevus prompt told the model that
low-grade corresponded to mild and high-grade to moderate-to-severe.
That binary is gone. This module replaces it with the actual v2.0
four-class schema.

Read this before changing anything here
---------------------------------------
Version 2.0 collapsed v1.0's five classes into four, specifically by
removing the standalone moderate-atypia class. Class I is low-grade
(mild-to-moderate) atypia; Class II is high-grade (high-end
moderate-to-severe) atypia.

That has a direct consequence for this study. The melanocytic arm
samples 50 mild, 50 moderate, 50 severe and 50 melanoma, but a
"moderate" case has no single correct v2.0 class: low-end moderate is
Class I, high-end moderate is Class II. The schema is built that way on
purpose, to stop forcing a split observers could not reproduce.

So this module does not pretend there is a 1:1 map. `expected_classes()`
returns a set, and the moderate stratum legitimately returns {I, II}.
The scorer counts a model answer as concordant if it lands in the
expected set, and `is_ambiguous()` flags those cases so they are
reported separately rather than quietly inflating agreement.

Do not "fix" this by collapsing moderate to one class. If the study needs
one class per case, it has to come from the reference dermatopathologist
assigning it per case into the ground truth's `mpath_dx_v2_reference`
column, not from a lookup table in code.
"""

from __future__ import annotations

CITATION = (
    "Barnhill RL, Elder DE, Piepkorn MW, et al. Revision of the "
    "Melanocytic Pathology Assessment Tool and Hierarchy for Diagnosis "
    "Classification Schema for Melanocytic Lesions: A Consensus "
    "Statement. JAMA Netw Open. 2023;6(1):e2250613. "
    "doi:10.1001/jamanetworkopen.2022.50613"
)

SCHEMA_VERSION = "MPATH-Dx v2.0"
CLASSES = ("0", "I", "II", "III", "IV")
BRESLOW_PT1B_CUTOFF_MM = 0.8

# Class II and above carries a re-excision recommendation. That binary is
# what actually changes management, and it is the third nevus scoring arm.
MANAGEMENT_THRESHOLD_CLASS = "II"

CLASS_DEFINITIONS = {
    "0": {
        "label": "Nondiagnostic",
        "definition": ("Unsatisfactory or nondiagnostic specimen; repeat "
                       "biopsy indicated."),
        "examples": [],
    },
    "I": {
        "label": "Low-grade atypia",
        "definition": ("Very low risk of continued proliferation and "
                       "progression to invasive melanoma. Melanocyte nuclei "
                       "less than 1.5 times the size of resting basal "
                       "keratinocyte nuclei."),
        "examples": ["common acquired nevus without atypia",
                     "congenital nevus without atypia",
                     "atypical/dysplastic nevus, low-grade atypia",
                     "common blue nevus"],
    },
    "II": {
        "label": "High-grade atypia",
        "definition": ("Low risk of progression to invasive melanoma; "
                       "re-excision with margins less than 1 cm indicated. "
                       "Melanocyte nuclei ranging from at least 1.5 times to "
                       "more than 2 times the size of resting basal "
                       "keratinocyte nuclei."),
        "examples": ["dysplastic nevus, high-grade atypia",
                     "Spitz nevus / tumour / melanocytoma",
                     "cellular blue nevus / melanocytoma",
                     "lentigo maligna", "melanoma in situ"],
    },
    "III": {
        "label": "Melanoma pT1a",
        "definition": ("Invasive melanoma, Breslow thickness less than "
                       "0.8 mm; relatively low risk of local and regional "
                       "metastasis."),
        "examples": ["invasive melanoma, pT1a"],
    },
    "IV": {
        "label": "Melanoma pT1b or greater",
        "definition": ("Invasive melanoma, Breslow thickness 0.8 mm or "
                       "greater; moderate to increased risk of regional or "
                       "distant metastasis."),
        "examples": ["invasive melanoma, pT1b or higher"],
    },
}

_ORDER = {c: i for i, c in enumerate(CLASSES)}


def _normalise(mpath_class: str) -> str:
    s = str(mpath_class).strip().upper()
    if s.startswith("CLASS"):
        s = s[5:].strip()
    return {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(s, s)


def class_rank(mpath_class: str) -> int:
    """Ordinal position of a class, for weighted agreement statistics."""
    key = _normalise(mpath_class)
    if key not in _ORDER:
        raise ValueError(f"unknown MPATH-Dx class: {mpath_class!r}")
    return _ORDER[key]


def is_valid_class(mpath_class: str) -> bool:
    return _normalise(mpath_class) in _ORDER


def expected_classes(stratum: str,
                     melanoma_subtype: str | None = None,
                     breslow_mm: float | None = None) -> set[str]:
    """MPATH-Dx v2.0 class(es) a case of this stratum should receive.

    Returns a set, because v2.0 does not resolve every case to one class
    from the stratum label alone. See the module docstring.

    `melanoma_subtype` is "in_situ" or "invasive"; `breslow_mm` is only
    consulted for invasive melanoma, where 0.8 mm splits III from IV.
    """
    s = str(stratum).strip().lower()

    if s == "mild":
        return {"I"}
    if s == "moderate":
        # Genuinely ambiguous under v2.0. Not a bug.
        return {"I", "II"}
    if s == "severe":
        return {"II"}
    if s == "melanoma":
        subtype = (melanoma_subtype or "").strip().lower()
        if subtype == "in_situ":
            # Melanoma in situ is Class II under v2.0, the same class as
            # high-grade dysplasia. This is the single most consequential
            # difference from v1.0 for the melanoma stratum: if the
            # stratum contains in situ cases, "melanoma" and "severe"
            # are not separable on class alone.
            return {"II"}
        if subtype == "invasive":
            if breslow_mm is None:
                return {"III", "IV"}
            return ({"III"} if float(breslow_mm) < BRESLOW_PT1B_CUTOFF_MM
                    else {"IV"})
        return {"II", "III", "IV"}

    raise ValueError(f"unknown melanocytic stratum: {stratum!r}")


def is_ambiguous(stratum: str, melanoma_subtype: str | None = None,
                 breslow_mm: float | None = None) -> bool:
    """True when the stratum label alone does not pin down one class."""
    return len(expected_classes(stratum, melanoma_subtype, breslow_mm)) > 1


def requires_reexcision(mpath_class: str) -> bool:
    """Class II and above carry a re-excision recommendation."""
    return class_rank(mpath_class) >= class_rank(MANAGEMENT_THRESHOLD_CLASS)


def prompt_block() -> str:
    """The v2.0 reference text injected into the nevus prompt.

    Lives here rather than in the analyzer so the definitions the model
    sees and the definitions the scorer applies cannot drift apart.
    """
    lines = [f"{SCHEMA_VERSION} classification schema (Barnhill et al., "
             "JAMA Netw Open 2023;6(1):e2250613):", ""]
    for cls in CLASSES:
        d = CLASS_DEFINITIONS[cls]
        lines.append(f"  Class {cls} - {d['label']}: {d['definition']}")
        if d["examples"]:
            lines.append(f"      Includes: {'; '.join(d['examples'])}.")
    lines += [
        "",
        "Note on version 2.0: it replaced the five-class version 1.0 schema "
        "and removed the standalone moderate-atypia class. Class I covers "
        "low-grade (mild-to-moderate) atypia and Class II covers high-grade "
        "(high-end moderate-to-severe) atypia. A lesion you grade as "
        "moderate dysplasia may therefore be Class I or Class II; decide "
        "from the cytologic criteria above, particularly nuclear size "
        "relative to resting basal keratinocytes, not from the word "
        "'moderate'.",
    ]
    return "\n".join(lines)
