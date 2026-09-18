"""Build the 350-case registry and the blinded reader manifest.

    python build_case_registry.py --scan slides/ --out data/case_registry.csv
    python build_case_registry.py --template          # empty, with headers
    python build_case_registry.py --reader-manifest   # blinded reader order

The registry is the single place that ties together the two arms of the
study:

  svs_path                the whole-slide file the dermatopathologists read
  tile_whole_slide/4x/10x/40x   the JPEGs the model reads
  tile_selection_method   curated or auto - who chose the fields, and
                          whether they could have seen the diagnosis
  reference_stratum       the ground-truth label, which never reaches a
                          model prompt

Reader manifest
---------------
`--reader-manifest` writes a separate CSV giving each reader a
independently shuffled case order under an opaque reading ID. Readers
should not receive cases grouped by stratum: 50 consecutive severe cases
tell a reader what to expect, and the resulting anchoring shows up as
inflated agreement. The shuffle is seeded so the order is reproducible
for the record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import random
import sys
from collections import Counter

import config


REGISTRY_FIELDS = [
    # reference_stratum is the MPATH-Dx v2.0 class (I/II/III/IV) on the
    # melanocytic side and the differentiation grade on the CSCC side.
    "case_id", "pathway", "reference_stratum",
    "lesion_category", "dysplasia_grade",
    "melanoma_subtype", "melanoma_histologic_subtype", "breslow_mm",
    "svs_path", "svs_sha256",
    "tile_whole_slide", "tile_4x", "tile_10x", "tile_40x",
    "tile_selection_method",
    "image_source_registry", "native_objective_power", "notes",
]

READER_FIELDS = ["reading_id", "reader_id", "sequence", "case_id",
                 "pathway", "svs_path"]


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    """Streamed hash: .svs files run to gigabytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_template(out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS).writeheader()
    print(f"Wrote empty registry template: {out}")
    print("Columns:\n  " + "\n  ".join(REGISTRY_FIELDS))


def scan(slide_dir: pathlib.Path, out: pathlib.Path,
         tile_root: pathlib.Path, hash_slides: bool) -> None:
    """Build registry rows from a directory of .svs files.

    Expects `<pathway>/<stratum>/<case>.svs`, e.g.
    slides/Nevus/severe/NEV-0123.svs
    """
    rows: list[dict] = []
    for svs in sorted(slide_dir.rglob("*.svs")):
        parts = svs.relative_to(slide_dir).parts
        if len(parts) < 3:
            print(f"  skipped (expected <pathway>/<stratum>/file.svs): {svs}")
            continue
        pathway, stratum = parts[0], parts[1]
        if pathway not in config.PATHWAYS:
            print(f"  skipped (unknown pathway {pathway!r}): {svs}")
            continue
        if stratum not in config.STRATA_BY_PATHWAY[pathway]:
            print(f"  skipped (unknown stratum {stratum!r}): {svs}")
            continue

        case_id = svs.stem
        tile_dir = tile_root / pathway / case_id
        row = {f: "" for f in REGISTRY_FIELDS}
        row.update({
            "case_id": case_id,
            "pathway": pathway,
            "reference_stratum": stratum,
            "svs_path": str(svs),
            "svs_sha256": sha256_file(svs) if hash_slides else "",
            "tile_selection_method": "auto",
        })
        for mag in config.MAGNIFICATIONS:
            candidate = tile_dir / f"{case_id}__{mag}.jpg"
            row[f"tile_{mag}"] = str(candidate) if candidate.exists() else ""
        rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}\n")
    report(rows)


def report(rows: list[dict]) -> None:
    """Print stratum counts against target, and tile completeness."""
    counts = Counter((r["pathway"], r["reference_stratum"]) for r in rows)
    print("Stratum counts (target "
          f"{config.N_PER_STRATUM} each, {config.TARGET_N_TOTAL} total):")
    total = 0
    for pathway in config.PATHWAYS:
        for stratum in config.STRATA_BY_PATHWAY[pathway]:
            n = counts.get((pathway, stratum), 0)
            total += n
            delta = n - config.N_PER_STRATUM
            mark = "ok" if delta == 0 else f"{delta:+d}"
            print(f"  {pathway:6} {stratum:10} {n:4}   {mark}")
    print(f"  {'TOTAL':17} {total:4}   "
          f"{'ok' if total == config.TARGET_N_TOTAL else f'{total - config.TARGET_N_TOTAL:+d}'}")

    incomplete = [r["case_id"] for r in rows
                  if not all(r.get(f"tile_{m}") for m in config.MAGNIFICATIONS)]
    if incomplete:
        print(f"\n{len(incomplete)} case(s) missing one or more tiles; "
              f"run extract_tiles.py. First few: "
              f"{', '.join(incomplete[:5])}")

    # Class II holds both high-grade dysplasia and melanoma in situ, and
    # the class label alone does not say which. Surface the mix so it is a
    # deliberate choice rather than an accident of whatever was on the shelf.
    class_two = [r for r in rows
                 if r.get("pathway") == "Nevus"
                 and r.get("reference_stratum") == "II"]
    if class_two:
        mix = Counter(r.get("lesion_category", "") or "unspecified"
                      for r in class_two)
        print(f"\nClass II composition ({len(class_two)} cases):")
        for category, count in sorted(mix.items()):
            print(f"  {category:24} {count:4}")
        print("  Class II covers high-grade dysplasia AND melanoma in situ. "
              "The class\n  label cannot distinguish them; melanoma_subtype "
              "is what does.")


def reader_manifest(registry: pathlib.Path, out: pathlib.Path,
                    readers: list[str], seed: int) -> None:
    with registry.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{registry} is empty")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=READER_FIELDS)
        writer.writeheader()
        for reader in readers:
            order = rows[:]
            random.Random(f"{seed}:{reader}").shuffle(order)
            for sequence, row in enumerate(order, start=1):
                writer.writerow({
                    "reading_id": hashlib.sha256(
                        f"{seed}:{reader}:{row['case_id']}".encode()
                    ).hexdigest()[:12],
                    "reader_id": reader,
                    "sequence": sequence,
                    "case_id": row["case_id"],
                    "pathway": row["pathway"],
                    "svs_path": row["svs_path"],
                })
    print(f"Wrote reader manifest for {len(readers)} reader(s) x "
          f"{len(rows)} cases to {out}")
    print("Give readers the reading_id and the .svs only. The case_id and "
          "pathway columns are for un-blinding after scoring.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", help="directory of <pathway>/<stratum>/*.svs")
    ap.add_argument("--out", default=config.CASE_REGISTRY)
    ap.add_argument("--tile-root", default=config.TILE_ROOT)
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip .svs hashing (much faster; loses provenance)")
    ap.add_argument("--reader-manifest", action="store_true")
    ap.add_argument("--readers", default="R1,R2,R3")
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    out = pathlib.Path(args.out)

    if args.template:
        write_template(out)
        return
    if args.reader_manifest:
        reader_manifest(out, pathlib.Path(config.READER_MANIFEST),
                        [r.strip() for r in args.readers.split(",") if r.strip()],
                        args.seed)
        return
    if args.scan:
        scan(pathlib.Path(args.scan), out, pathlib.Path(args.tile_root),
             not args.no_hash)
        return

    if out.exists():
        with out.open(newline="", encoding="utf-8") as fh:
            report(list(csv.DictReader(fh)))
    else:
        ap.error("pass --scan, --template or --reader-manifest")


if __name__ == "__main__":
    main()
