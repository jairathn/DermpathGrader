"""Offline smoke tests. No API key and no network required.

    python -m pytest tests/ -q      (or: python tests/test_smoke.py)

These cover the parts of the v2 protocol that are cheap to get wrong and
expensive to discover after a 350-case batch: the magnification contract,
the MPATH-Dx v2.0 mapping, image determinism, and the verifier's ability
to notice a log that does not describe what was sent.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import config
import image_utils
import mpath_dx
from grading_logger import CaseLogger


def _fake_case(size=(1200, 900)) -> dict:
    directory = pathlib.Path(tempfile.mkdtemp())
    sources = {}
    for index, mag in enumerate(config.MAGNIFICATIONS):
        path = directory / f"{mag}.png"
        rng = np.random.default_rng(index)
        Image.fromarray(
            rng.integers(0, 255, (size[1], size[0], 3), dtype="uint8")
        ).save(path)
        sources[mag] = path
    return sources


def test_study_design_totals():
    assert config.N_PER_STRATUM == 50
    assert config.TARGET_N["Nevus"] == 200
    assert config.TARGET_N["CSCC"] == 150
    assert config.TARGET_N_TOTAL == 350
    # The melanocytic strata are the MPATH-Dx v2.0 classes themselves.
    assert config.NEVUS_STRATA == ("I", "II", "III", "IV")


def test_mpath_v2_classes_are_the_strata():
    for stratum in config.NEVUS_STRATA:
        assert mpath_dx.expected_class(stratum) == stratum
    assert mpath_dx.expected_class("Class III") == "III"
    # A legacy three-tier label must fail loudly rather than be guessed at.
    try:
        mpath_dx.expected_class("moderate")
    except ValueError:
        pass
    else:
        raise AssertionError("legacy labels must not resolve to a class")
    assert not mpath_dx.requires_reexcision("I")
    assert mpath_dx.requires_reexcision("II")
    assert "e2250613" in mpath_dx.CITATION


def test_melanoma_class_assignment():
    # In situ melanoma is Class II, the same class as high-grade dysplasia.
    assert mpath_dx.class_for_melanoma("in_situ") == "II"
    assert mpath_dx.class_for_melanoma("invasive", 0.4) == "III"
    assert mpath_dx.class_for_melanoma("invasive", 0.79) == "III"
    assert mpath_dx.class_for_melanoma("invasive", 0.8) == "IV"
    assert mpath_dx.class_for_melanoma("invasive", 3.0) == "IV"
    # Invasive melanoma without a thickness cannot be placed.
    try:
        mpath_dx.class_for_melanoma("invasive")
    except ValueError:
        pass
    else:
        raise AssertionError("invasive melanoma needs a Breslow thickness")
    # Class II is not an invasive-melanoma class, because it also holds
    # high-grade dysplasia.
    assert not mpath_dx.is_melanoma_class("II")
    assert mpath_dx.is_melanoma_class("III")
    assert mpath_dx.is_melanoma_class("IV")


def test_legacy_three_tier_mapping_stays_honest():
    assert mpath_dx.class_from_dysplasia_grade("mild") == {"I"}
    # v2.0 deleted the standalone moderate class, so this spans two.
    assert mpath_dx.class_from_dysplasia_grade("moderate") == {"I", "II"}
    assert mpath_dx.is_ambiguous("moderate")
    assert mpath_dx.class_from_dysplasia_grade("severe") == {"II"}
    assert not mpath_dx.is_ambiguous("severe")


def test_all_four_magnifications_required():
    sources = _fake_case()
    prepared = image_utils.prepare_case_images(sources)
    assert [p.magnification for p in prepared] == list(config.MAGNIFICATIONS)

    partial = {k: v for k, v in sources.items() if k != "40x"}
    try:
        image_utils.prepare_case_images(partial)
    except image_utils.ImagePreparationError:
        pass
    else:
        raise AssertionError("a 3-image case must not be gradeable")


def test_image_preprocessing_is_deterministic_and_bounded():
    sources = _fake_case(size=(4000, 3000))
    first = image_utils.prepare_case_images(sources)
    second = image_utils.prepare_case_images(sources)
    assert [p.sent_sha256 for p in first] == [p.sent_sha256 for p in second]
    for image in first:
        assert max(image.sent_dimensions_px) <= config.MAX_IMAGE_EDGE_PX
        assert image.sent_bytes <= config.MAX_IMAGE_BYTES
        assert (config.JPEG_QUALITY_FLOOR
                <= image.compression_quality
                <= config.JPEG_QUALITY_START)
        assert image.sent_media_type == config.OUTBOUND_MEDIA_TYPE


def test_svs_is_rejected_before_the_api():
    try:
        image_utils.prepare_image(pathlib.Path("nonexistent.svs"), "4x")
    except (image_utils.ImagePreparationError, FileNotFoundError):
        pass
    else:
        raise AssertionError(".svs must never reach the vision API")


def test_envelope_matches_the_images_sent():
    prepared = image_utils.prepare_case_images(_fake_case())
    structure = image_utils.message_structure(prepared, "0" * 64)
    blocks = structure[0]["blocks"]
    images = [b for b in blocks if b["type"] == "image"]
    assert len(images) == len(config.MAGNIFICATIONS)
    assert [b["magnification"] for b in images] == list(config.MAGNIFICATIONS)
    assert [b["sha256"] for b in images] == [p.sent_sha256 for p in prepared]
    assert blocks[-1]["role"] == "prompt"


def test_logger_records_four_images_and_refuses_overwrite():
    prepared = image_utils.prepare_case_images(_fake_case())
    CaseLogger.LOG_ROOT = pathlib.Path(tempfile.mkdtemp())
    logger = CaseLogger("Nevus", "T-001", 1, "sess")
    logger.set_image_set(prepared, tile_selection_method="curated")
    logger.set_request(model=config.MODEL_ID, max_tokens=config.MAX_TOKENS,
                       system_text="", user_text="p",
                       message_structure=image_utils.message_structure(
                           prepared, hashlib.sha256(b"p").hexdigest()),
                       thinking=config.THINKING, effort=config.EFFORT,
                       schema_enforced=True, output_schema_sha256="c" * 64)
    path = logger.save()
    record = json.loads(path.read_text())
    assert record["log_version"] == config.LOG_VERSION
    assert len(record["images"]) == len(config.MAGNIFICATIONS)
    assert record["request"]["temperature"] is None
    try:
        logger.save()
    except FileExistsError:
        pass
    else:
        raise AssertionError("logs must never be overwritten")


def test_analyzer_schemas_cover_the_study_labels():
    from nevi_analyzer import NEVUS_OUTPUT_SCHEMA
    from image_analyzer import CSCC_OUTPUT_SCHEMA

    properties = NEVUS_OUTPUT_SCHEMA["properties"]
    classes = properties["mpath_dx_v2_class"]["enum"]
    assert classes == list(mpath_dx.CLASSES)
    # Every sampled stratum must be an emittable answer.
    assert set(config.NEVUS_STRATA) <= set(classes)
    assert properties["melanoma_subtype"]["enum"] == list(
        config.MELANOMA_SUBTYPES)
    assert "in_situ" in properties["melanoma_subtype"]["enum"]
    assert "invasive" in properties["melanoma_subtype"]["enum"]
    assert properties["melanoma_histologic_subtype"]["enum"] == list(
        config.MELANOMA_HISTOLOGIC_SUBTYPES)
    grades = CSCC_OUTPUT_SCHEMA["properties"]["primary_grade"]["enum"]
    assert len(grades) == len(config.CSCC_STRATA)


def test_consistency_flags_catch_a_contradictory_answer():
    from nevi_analyzer import NeviAnalyzer
    flag = NeviAnalyzer._derive_consistency_flags

    def make(**overrides):
        base = {"mpath_dx_v2_class": "I",
                "lesion_category": "dysplastic_nevus",
                "dysplasia_grade": "mild",
                "melanoma_subtype": "not_applicable",
                "melanoma_histologic_subtype": "not_applicable",
                "breslow_estimate_mm": None}
        base.update(overrides)
        return flag(base)

    assert make()["internally_consistent"]

    # Invasive melanoma at 1.1 mm is Class IV.
    assert make(mpath_dx_v2_class="IV", lesion_category="melanoma",
                dysplasia_grade="not_applicable",
                melanoma_subtype="invasive",
                melanoma_histologic_subtype="nodular",
                breslow_estimate_mm=1.1)["internally_consistent"]

    # Melanoma in situ is Class II; calling it Class III is a contradiction.
    assert make(mpath_dx_v2_class="II", lesion_category="melanoma",
                dysplasia_grade="not_applicable",
                melanoma_subtype="in_situ",
                melanoma_histologic_subtype="lentigo_maligna"
                )["internally_consistent"]
    assert not make(mpath_dx_v2_class="III", lesion_category="melanoma",
                    dysplasia_grade="not_applicable",
                    melanoma_subtype="in_situ",
                    melanoma_histologic_subtype="lentigo_maligna"
                    )["internally_consistent"]

    # A thin melanoma placed in Class IV contradicts its own Breslow.
    assert not make(mpath_dx_v2_class="IV", lesion_category="melanoma",
                    dysplasia_grade="not_applicable",
                    melanoma_subtype="invasive",
                    melanoma_histologic_subtype="nodular",
                    breslow_estimate_mm=0.4)["internally_consistent"]

    # Moderate dysplasia is legitimately Class I or Class II under v2.0.
    for cls in ("I", "II"):
        assert make(mpath_dx_v2_class=cls,
                    dysplasia_grade="moderate")["internally_consistent"]

    # A nevus cannot sit in an invasive-melanoma class.
    assert not make(mpath_dx_v2_class="III",
                    dysplasia_grade="severe")["internally_consistent"]


def test_scoring_statistics():
    import join_and_score as scoring

    labels = list(config.NEVUS_STRATA)   # I, II, III, IV
    perfect = [(x, x) for x in labels] * 5
    assert abs(scoring.cohen_kappa(perfect, labels) - 1.0) < 1e-9
    assert abs(scoring.weighted_kappa(perfect, labels) - 1.0) < 1e-9
    low, high = scoring.wilson(50, 50)
    assert high == 1.0 and low < 1.0
    # Quadratic weighting must give partial credit for an adjacent miss.
    adjacent = [("III", "II")] * 10 + [("I", "I")] * 10
    assert (scoring.weighted_kappa(adjacent, labels)
            > scoring.cohen_kappa(adjacent, labels))


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
