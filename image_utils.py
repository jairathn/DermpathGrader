"""Image preprocessing and multi-image payload construction (protocol v2.0).

What changed from v1
--------------------
1. A case is now a *set* of four images (whole slide, 4x, 10x, 40x)
   rather than one. `prepare_case_images()` is the single entry point and
   returns one `PreparedImage` per magnification, in config order.

2. There is one preprocessing path. `prepare_image()` does the work, and
   both the Streamlit uploader and the batch runner call it. Previously
   `validate_and_process_image()` and `process_image_file()` were near
   duplicates that returned different shapes, which made the UI and batch
   results only incidentally comparable (CLAUDE.md landmine 5).

3. A long-edge cap (config.MAX_IMAGE_EDGE_PX) is applied. v1 had none, so
   a full-resolution frame was uploaded and then downsampled server-side:
   paid for, never used. At four images per case that waste is 4x.

4. `compression_quality` is recorded for real. v1 logged `None` because
   the quality loop did not expose its internals, so the logs could not
   say what the model actually saw.

5. Everything is re-encoded to JPEG, deterministically. v1 preserved PNG
   for RGBA inputs and JPEG otherwise, so two cases could reach the model
   through different codecs. Histology is H&E; there is no alpha channel
   worth preserving, and one codec makes cases comparable.

Streamlit is imported lazily so the batch runner does not need to
monkey-patch `st.*` to stay quiet, as run_tests.py previously did.
"""

from __future__ import annotations

import base64
import hashlib
import io
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from PIL import Image

import config


# ── data model ───────────────────────────────────────────────────────

@dataclass
class PreparedImage:
    """One magnification of one case, ready to send."""

    magnification: str
    source_path: str
    source_filename: str
    source_sha256: str
    source_dimensions_px: list[int]
    sent_media_type: str
    sent_dimensions_px: list[int]
    sent_bytes: int
    sent_sha256: str
    resize_applied: bool
    compression_quality: int
    b64: str = field(repr=False, default="")

    def to_log_dict(self) -> dict[str, Any]:
        """The per-image record written into the case log.

        `b64` is deliberately excluded: logs stay diffable and small.
        """
        return {
            "magnification": self.magnification,
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "source_dimensions_px": self.source_dimensions_px,
            "sent_media_type": self.sent_media_type,
            "sent_dimensions_px": self.sent_dimensions_px,
            "sent_bytes": self.sent_bytes,
            "sent_sha256": self.sent_sha256,
            "resize_applied": self.resize_applied,
            "compression_quality": self.compression_quality,
        }


class ImagePreparationError(RuntimeError):
    """Raised when an image cannot be brought under the API limits."""


# ── core preprocessing ───────────────────────────────────────────────

def _to_rgb(image: Image.Image) -> Image.Image:
    """Flatten to RGB. H&E has no alpha worth keeping."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def _cap_long_edge(image: Image.Image,
                   max_edge: int = config.MAX_IMAGE_EDGE_PX
                   ) -> tuple[Image.Image, bool]:
    """Downscale so the long edge is at most `max_edge`. Never upscales."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image, False
    scale = max_edge / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), True


def _encode_under_limit(image: Image.Image,
                        max_bytes: int = config.MAX_IMAGE_BYTES
                        ) -> tuple[bytes, int]:
    """JPEG-encode, stepping quality down until it fits.

    Returns (bytes, quality_used). Raises rather than dropping below
    config.JPEG_QUALITY_FLOOR: below that, 40x cytology develops
    compression artefacts that are indistinguishable from nuclear
    detail, and a grade read off such an image is not trustworthy.
    """
    quality = config.JPEG_QUALITY_START
    last_size = None
    while quality >= config.JPEG_QUALITY_FLOOR:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        last_size = len(data)
        if last_size <= max_bytes:
            return data, quality
        quality -= config.JPEG_QUALITY_STEP

    raise ImagePreparationError(
        f"cannot fit image under {max_bytes} bytes at quality "
        f">= {config.JPEG_QUALITY_FLOOR} (smallest was {last_size} bytes). "
        f"Re-export this tile at a smaller pixel size rather than letting "
        f"the encoder degrade it further."
    )


def prepare_image(source: str | pathlib.Path | bytes,
                  magnification: str,
                  source_filename: str | None = None) -> PreparedImage:
    """Preprocess one image into an API-ready payload plus its metadata.

    `source` is a path, or raw bytes (as from a Streamlit upload). This
    is the only preprocessing path in the codebase; both the UI and the
    batch runner go through it, so a case graded interactively and the
    same case graded in batch send byte-identical images.
    """
    if isinstance(source, (str, pathlib.Path)):
        path = pathlib.Path(source)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        if path.suffix.lower() in config.WSI_SUFFIXES:
            raise ImagePreparationError(
                f"{path.name} is a whole-slide image. The vision API cannot "
                f"read .svs. Run extract_tiles.py to generate the "
                f"magnification JPEGs first."
            )
        raw = path.read_bytes()
        display_name = source_filename or path.name
        source_path = str(path)
    else:
        raw = bytes(source)
        display_name = source_filename or "<upload>"
        source_path = ""

    source_sha = hashlib.sha256(raw).hexdigest()

    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        source_dims = list(opened.size)
        rgb = _to_rgb(opened)

    resized, resize_applied = _cap_long_edge(rgb)
    data, quality = _encode_under_limit(resized)

    return PreparedImage(
        magnification=magnification,
        source_path=source_path,
        source_filename=display_name,
        source_sha256=source_sha,
        source_dimensions_px=source_dims,
        sent_media_type=config.OUTBOUND_MEDIA_TYPE,
        sent_dimensions_px=list(resized.size),
        sent_bytes=len(data),
        sent_sha256=hashlib.sha256(data).hexdigest(),
        resize_applied=resize_applied,
        compression_quality=quality,
        b64=base64.b64encode(data).decode("ascii"),
    )


# ── case-level assembly ──────────────────────────────────────────────

def prepare_case_images(sources: dict[str, Any],
                        required: Iterable[str] | None = None
                        ) -> list[PreparedImage]:
    """Preprocess a whole case: one image per required magnification.

    `sources` maps magnification -> path or bytes. Returns the prepared
    images in config.MAGNIFICATIONS order, which is the order they are
    presented to the model and recorded in the log.

    A missing magnification is a hard error, not a warning. A case graded
    on three images is not comparable to one graded on four, and silently
    allowing it would put non-comparable cases in the same analysis.
    """
    required = tuple(required) if required is not None \
        else config.REQUIRED_MAGNIFICATIONS

    missing = [m for m in required if not sources.get(m)]
    if missing:
        raise ImagePreparationError(
            f"missing required magnification(s): {', '.join(missing)}. "
            f"All of {', '.join(required)} must be present for a case to be "
            f"gradeable under protocol v{config.PROTOCOL_VERSION}."
        )

    ordered = [m for m in config.MAGNIFICATIONS if m in required]
    prepared = [prepare_image(sources[m], m) for m in ordered]

    total = sum(p.sent_bytes for p in prepared)
    # The API caps the whole request at 32 MB; base64 expands ~4/3.
    if total * 4 / 3 > 30_000_000:
        raise ImagePreparationError(
            f"combined payload {total} bytes exceeds the per-request budget "
            f"once base64-encoded. Lower config.MAX_IMAGE_BYTES."
        )
    return prepared


def build_image_content_blocks(prepared: list[PreparedImage]
                               ) -> list[dict[str, Any]]:
    """Interleave caption + image blocks for the API message.

    Each image is preceded by a text block naming its magnification and
    what it is for. Unlabelled images in a multi-image request get
    conflated; the caption is what lets the model attribute a finding to
    a power, which is in turn what makes `magnification_evidence` in the
    response meaningful.
    """
    blocks: list[dict[str, Any]] = []
    total = len(prepared)
    for index, image in enumerate(prepared, start=1):
        caption = config.MAGNIFICATION_CAPTIONS.get(
            image.magnification, image.magnification)
        blocks.append({
            "type": "text",
            "text": f"Image {index} of {total} - {caption}",
        })
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.sent_media_type,
                "data": image.b64,
            },
        })
    return blocks


def message_structure(prepared: list[PreparedImage],
                      prompt_sha256: str) -> list[dict[str, Any]]:
    """The `request.message_structure` record for the case log.

    Mirrors the real block order so the verifier can confirm that what
    was logged is what was sent.
    """
    blocks: list[dict[str, Any]] = []
    total = len(prepared)
    for index, image in enumerate(prepared, start=1):
        caption = config.MAGNIFICATION_CAPTIONS.get(
            image.magnification, image.magnification)
        blocks.append({
            "type": "text",
            "role": "caption",
            "magnification": image.magnification,
            "sha256": hashlib.sha256(
                f"Image {index} of {total} - {caption}".encode()).hexdigest(),
        })
        blocks.append({
            "type": "image",
            "magnification": image.magnification,
            "media_type": image.sent_media_type,
            "sha256": image.sent_sha256,
        })
    blocks.append({"type": "text", "role": "prompt", "sha256": prompt_sha256})
    return [{"role": "user", "blocks": blocks}]


# ── Streamlit-facing helpers ─────────────────────────────────────────

def validate_upload(uploaded_file) -> str | None:
    """Return an error string, or None if the upload looks usable."""
    if uploaded_file is None:
        return "No file uploaded"
    if uploaded_file.size > 60 * 1024 * 1024:
        return ("File too large. Upload an exported tile, not a whole-slide "
                "scan.")
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
    if suffix in config.WSI_SUFFIXES:
        return ("Whole-slide .svs files cannot be sent to the vision API. "
                "Run extract_tiles.py to produce the four magnification "
                "JPEGs, then upload those.")
    if suffix not in config.ACCEPTED_IMAGE_SUFFIXES:
        return (f"Unsupported file type {suffix}. Accepted: "
                f"{', '.join(config.ACCEPTED_IMAGE_SUFFIXES)}")
    return None
