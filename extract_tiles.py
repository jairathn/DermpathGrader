"""Extract the four magnification JPEGs from a whole-slide .svs file.

    python extract_tiles.py --registry data/case_registry.csv
    python extract_tiles.py --svs slides/NEV-001.svs --case-id NEV-001 \
                            --pathway Nevus

Why this exists
---------------
Dermatopathologists read the `.svs` directly. The vision API cannot open
one, so the model arm reads JPEG derivatives. This script is the bridge,
and it is the point where the two arms stop seeing the same thing.

The field-of-view problem - read before running a batch
-------------------------------------------------------
A human reading a WSI pans and zooms freely: they choose where to look at
40x *after* seeing the whole slide, and they can go back. The model gets
four fixed frames. Whoever picks the 4x/10x/40x coordinates therefore
makes part of the diagnostic decision before the model ever sees the
case, and if that person knows the diagnosis, the label leaks into the
input.

This is the single largest threat to the validity of the model arm. It is
not solvable in code, so the script does two things instead:

  1. `--source curated` (default) reads coordinates from
     data/tile_coords.csv, which a person fills in. Record who chose them
     and whether they were blinded, in that file's `chosen_by` and
     `blinded` columns.

  2. `--source auto` picks the densest tissue region by a fixed
     deterministic rule, with no human in the loop. Nobody sees the
     diagnosis, so nothing leaks, but the frame may miss the diagnostic
     area entirely - which is itself a finding worth reporting rather
     than a bug to patch.

Either way `tile_selection_method` is written into the registry and into
every case log, so the manuscript can state which was used. Do not mix
methods within a stratum: that confounds selection method with grade.

openslide
---------
Requires `openslide-python` and the OpenSlide C library, which is a
system package (`apt install libopenslide0`, `brew install openslide`).
It is an optional dependency: the rest of the codebase runs without it,
and if the tiles are exported by other software (Aperio ImageScope, QuPath)
this script is not needed at all - just populate the registry paths.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import config

try:
    import openslide
except ImportError:  # pragma: no cover - optional dependency
    openslide = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


TILE_PX = 2048  # square region read from the slide, before the long-edge cap


def _require_openslide() -> None:
    if openslide is None:
        sys.exit(
            "openslide-python is not installed.\n"
            "  pip install openslide-python\n"
            "and install the OpenSlide C library "
            "(apt install libopenslide0 / brew install openslide).\n\n"
            "If your tiles are already exported from ImageScope or QuPath, "
            "you do not need this script: put their paths in "
            f"{config.CASE_REGISTRY} instead."
        )


def native_objective_power(slide) -> float:
    """Scanning objective power, e.g. 20.0 or 40.0."""
    raw = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if raw is None:
        raise ValueError(
            "slide does not declare objective power; pass --native-power "
            "explicitly after checking the scanner's metadata"
        )
    return float(raw)


def _level_and_scale(slide, target_power: float, native_power: float):
    """Pick the pyramid level closest to (but not below) the target power.

    Reading from the nearest level and resizing down is sharper and far
    cheaper than reading level 0 and downsampling a huge region.
    """
    wanted_downsample = native_power / target_power
    best_level = 0
    for level in range(slide.level_count):
        if slide.level_downsamples[level] <= wanted_downsample * 1.001:
            best_level = level
    residual = wanted_downsample / slide.level_downsamples[best_level]
    return best_level, residual


def whole_slide_thumbnail(slide, max_edge: int = config.MAX_IMAGE_EDGE_PX):
    """Overview of the entire specimen."""
    return slide.get_thumbnail((max_edge, max_edge)).convert("RGB")


def auto_center(slide) -> tuple[int, int]:
    """Deterministic tissue centroid, in level-0 coordinates.

    Thresholds the thumbnail on saturation, which separates stained
    tissue from background glass far better than luminance does, then
    returns the centroid of the tissue mask. No human, no label leak.
    """
    if np is None:
        raise RuntimeError("numpy is required for --source auto")
    thumb = slide.get_thumbnail((1024, 1024)).convert("HSV")
    arr = np.asarray(thumb)
    saturation = arr[:, :, 1]
    mask = saturation > max(20, int(saturation.mean()))
    if not mask.any():
        mask = np.ones_like(saturation, dtype=bool)
    ys, xs = np.nonzero(mask)
    w0, h0 = slide.level_dimensions[0]
    cx = int(xs.mean() / arr.shape[1] * w0)
    cy = int(ys.mean() / arr.shape[0] * h0)
    return cx, cy


def read_region_at_power(slide, center_xy, target_power, native_power,
                         tile_px=TILE_PX):
    """Read a square field centred on `center_xy` at `target_power`."""
    level, residual = _level_and_scale(slide, target_power, native_power)
    read_px = int(round(tile_px * residual))
    downsample = slide.level_downsamples[level]

    # Region origin is level-0 coordinates even when reading a higher level.
    half_level0 = int(read_px * downsample / 2)
    x0 = max(0, center_xy[0] - half_level0)
    y0 = max(0, center_xy[1] - half_level0)

    region = slide.read_region((x0, y0), level, (read_px, read_px))
    region = region.convert("RGB")
    if read_px != tile_px:
        from PIL import Image
        region = region.resize((tile_px, tile_px), Image.Resampling.LANCZOS)
    return region


def load_coords(path: str) -> dict[str, dict[str, tuple[int, int]]]:
    """case_id -> {magnification: (x, y)} from the curated coords CSV."""
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, tuple[int, int]]] = {}
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            case = row["case_id"].strip()
            mag = row["magnification"].strip()
            out.setdefault(case, {})[mag] = (int(row["x"]), int(row["y"]))
    return out


def extract_case(svs_path: pathlib.Path, case_id: str, pathway: str,
                 out_root: pathlib.Path, source: str,
                 coords: dict[str, tuple[int, int]] | None,
                 native_power_override: float | None = None) -> dict:
    _require_openslide()
    slide = openslide.OpenSlide(str(svs_path))
    try:
        native = native_power_override or native_objective_power(slide)
        out_dir = out_root / pathway / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        written: dict[str, str] = {}
        for mag in config.MAGNIFICATIONS:
            if mag == "whole_slide":
                image = whole_slide_thumbnail(slide)
            else:
                power = float(mag.rstrip("x"))
                if power > native:
                    raise ValueError(
                        f"{svs_path.name} was scanned at {native}x; a {mag} "
                        f"frame would be upsampled, not resolved. Either "
                        f"rescan or drop {mag} from config.MAGNIFICATIONS "
                        f"for the whole study - not for this case alone."
                    )
                if source == "curated":
                    if not coords or mag not in coords:
                        raise ValueError(
                            f"no curated coordinates for {case_id} {mag} in "
                            f"{config.TILE_COORDS}; add them or rerun with "
                            f"--source auto"
                        )
                    center = coords[mag]
                else:
                    center = auto_center(slide)
                image = read_region_at_power(slide, center, power, native)

            dest = out_dir / f"{case_id}__{mag}.jpg"
            image.save(dest, format="JPEG", quality=95, optimize=True)
            written[mag] = str(dest)

        return {
            "case_id": case_id,
            "pathway": pathway,
            "native_objective_power": native,
            "tile_selection_method": source,
            **{f"tile_{m}": p for m, p in written.items()},
        }
    finally:
        slide.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", help="case_registry.csv to process in bulk")
    ap.add_argument("--svs", help="single .svs file")
    ap.add_argument("--case-id")
    ap.add_argument("--pathway", choices=list(config.PATHWAYS))
    ap.add_argument("--out-root", default=config.TILE_ROOT)
    ap.add_argument("--source", choices=["curated", "auto"], default="curated",
                    help="curated reads data/tile_coords.csv; auto uses the "
                         "deterministic tissue centroid (no label leak, but "
                         "may miss the diagnostic field)")
    ap.add_argument("--native-power", type=float,
                    help="override the slide's declared objective power")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    coords_all = load_coords(config.TILE_COORDS)

    if args.svs:
        if not (args.case_id and args.pathway):
            sys.exit("--svs requires --case-id and --pathway")
        result = extract_case(pathlib.Path(args.svs), args.case_id,
                              args.pathway, out_root, args.source,
                              coords_all.get(args.case_id),
                              args.native_power)
        print(result)
        return

    if not args.registry:
        sys.exit("pass --registry or --svs")

    rows = list(csv.DictReader(
        pathlib.Path(args.registry).open(newline="", encoding="utf-8")))
    done = failed = 0
    for row in rows:
        svs = row.get("svs_path", "").strip()
        if not svs:
            continue
        try:
            extract_case(pathlib.Path(svs), row["case_id"], row["pathway"],
                         out_root, args.source,
                         coords_all.get(row["case_id"]), args.native_power)
            done += 1
            print(f"  {row['case_id']} ok")
        except Exception as exc:
            failed += 1
            print(f"  {row['case_id']} FAILED: {exc}")

    print(f"\n{done} extracted, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
