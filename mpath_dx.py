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

The study samples these classes directly
---------------------------------------
As of 2026-09-18 the melanocytic strata ARE the v2.0 classes: 50 each of
Class I, II, III and IV. That resolves the problem the earlier three-tier
design had.

v2.0 produced its four classes by removing v1.0's standalone
moderate-atypia category - Class I is "mild-to-moderate" atypia and Class
II is "high-end moderate-to-severe". A case labelled "moderate dysplasia"
therefore had no single correct class, and scoring it meant either
accepting both answers or inventing a split the consensus panel had
deliberately abandoned. Sampling the classes directly removes that
entirely: every case has exactly one correct class, assigned by the
reference dermatopathologist.

`expected_class()` is now the scoring entry point and returns one class.
`class_from_dysplasia_grade()` remains for the secondary three-tier
field and for converting legacy labels during case selection; it still
returns a set, and still returns {I, II} for "moderate", because that
ambiguity is a real property of the schema and not something to paper
over when mapping old labels in.

One thing to keep in view when composing Class II
--------------------------------------------------
Class II contains both high-grade dysplastic nevi and melanoma in situ.
They are the same class by design. The `melanoma_subtype` field is what
separates them, so a Class II stratum that is entirely melanoma in situ
(or entirely dysplasia) will not be detectable from the class label
alone. Compose it deliberately and record the mix.
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


def expected_class(stratum: str) -> str:
    """The single correct class for a case in this stratum.

    The melanocytic strata are the v2.0 classes, so this is a validation
    and normalisation step rather than a mapping. It raises on anything
    that is not a class, which is what catches a registry still carrying
    legacy mild/moderate/severe labels.
    """
    key = _normalise(stratum)
    if key not in _ORDER:
        raise ValueError(
            f"{stratum!r} is not an MPATH-Dx v2.0 class. The melanocytic "
            f"strata are {', '.join(CLASSES)}. If this is a legacy "
            f"three-tier label, convert it with "
            f"class_from_dysplasia_grade() and have a dermatopathologist "
            f"resolve any case that maps to more than one class."
        )
    return key


def class_from_dysplasia_grade(
        grade: str,
        melanoma_subtype: str | None = None,
        breslow_mm: float | None = None) -> set[str]:
    """Map a legacy three-tier label to the v2.0 class(es) it could be.

    Used when importing cases labelled on the old scale, and for the
    secondary descriptive dysplasia field. Returns a SET because the
    mapping is genuinely not one-to-one: "moderate" spans Class I and
    Class II under v2.0. A case that lands here with two classes needs a
    dermatopathologist to pick one before it can enter the study.
    """
    s = str(grade).strip().lower()

    if s == "mild":
        return {"I"}
    if s == "moderate":
        # Genuinely ambiguous under v2.0. Not a bug, and not resolvable
        # from the word alone.
        return {"I", "II"}
    if s == "severe":
        return {"II"}
    if s == "melanoma":
        subtype = (melanoma_subtype or "").strip().lower()
        if subtype == "in_situ":
            # Melanoma in situ is Class II, alongside high-grade
            # dysplasia. The subtype field is what tells them apart.
            return {"II"}
        if subtype == "invasive":
            if breslow_mm is None:
                return {"III", "IV"}
            return ({"III"} if float(breslow_mm) < BRESLOW_PT1B_CUTOFF_MM
                    else {"IV"})
        return {"II", "III", "IV"}

    raise ValueError(f"unknown dysplasia grade: {grade!r}")


def class_for_melanoma(melanoma_subtype: str,
                       breslow_mm: float | None = None) -> str:
    """Class for a melanoma, given its subtype and thickness.

    in situ -> II; invasive <0.8 mm -> III; invasive >=0.8 mm -> IV.
    Raises when invasive melanoma arrives without a thickness, because
    III and IV cannot be separated without one.
    """
    subtype = str(melanoma_subtype).strip().lower()
    if subtype == "in_situ":
        return "II"
    if subtype == "invasive":
        if breslow_mm is None:
            raise ValueError(
                "invasive melanoma needs a Breslow thickness to separate "
                "Class III (<0.8 mm) from Class IV (>=0.8 mm)")
        return ("III" if float(breslow_mm) < BRESLOW_PT1B_CUTOFF_MM
                else "IV")
    raise ValueError(f"unknown melanoma subtype: {melanoma_subtype!r}")


def is_melanoma_class(mpath_class: str) -> bool:
    """True for the invasive-melanoma classes.

    Class II is excluded deliberately: it holds melanoma in situ AND
    high-grade dysplasia, so the class alone does not establish melanoma.
    Use the melanoma_subtype field for that.
    """
    return _normalise(mpath_class) in ("III", "IV")


def is_ambiguous(grade: str, melanoma_subtype: str | None = None,
                 breslow_mm: float | None = None) -> bool:
    """True when a LEGACY three-tier label does not pin down one class.

    Only meaningful for imports. Study strata are classes and are never
    ambiguous.
    """
    return len(class_from_dysplasia_grade(
        grade, melanoma_subtype, breslow_mm)) > 1


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
