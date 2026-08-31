"""The metadata tool: everything about an image file that is not a tag.

The tagging agent reads pixels and answers in the controlled vocabulary. It
never reports how wide the image is, what camera shot it, or whether this file
is the same picture as one fetched last week -- none of that is a marketing
tag, and asking a model for it would be paying for facts the file already
states. So this tool reads them directly.

Two things it produces are load-bearing elsewhere:

* `phash` -- a 64-bit perceptual hash. Sony serves the same product shot from
  several URLs and at several sizes, so byte equality catches almost nothing;
  a perceptual hash catches the re-encodes too. `scraper` asks the database for
  the hashes it already holds and skips those downloads, which is the only
  reason a nightly fetch does not re-tag the whole catalogue.
* `category` / `product` -- read off the folder path, the same way
  `agent.graph._infer_context` reads them, so an uploaded folder and a fetched
  one describe themselves identically.

Pillow does all of it, so this tool adds no dependency of its own.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass

from PIL import ExifTags, Image

from agent.graph import _infer_context, _split_context
from agent.logconf import get_logger

log = get_logger(__name__)

MIME_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
              "GIF": "image/gif", "TIFF": "image/tiff"}

# The EXIF fields a content team plausibly cares about. The rest -- maker
# notes, GPS, thumbnails -- is noise in a results table, and marketing images
# have usually had it stripped anyway.
_EXIF_WANTED = {"DateTimeOriginal": "shot_at", "DateTime": "shot_at",
                "Make": "camera_make", "Model": "camera_model",
                "Software": "software", "Orientation": "orientation"}

_EXIF_NAMES = {tag: name for tag, name in ExifTags.TAGS.items()}


@dataclass
class ImageMeta:
    """One image file, as the database records it."""

    path: str
    name: str = ""
    category: str = ""
    product: str = ""
    fmt: str = ""
    mime: str = ""
    width: int = 0
    height: int = 0
    bytes: int = 0
    mode: str = ""
    sha256: str = ""
    phash: str = ""
    shot_at: str = ""
    camera_make: str = ""
    camera_model: str = ""
    software: str = ""
    orientation: int = 0

    @property
    def megapixels(self) -> float:
        return round(self.width * self.height / 1_000_000, 2)

    def as_dict(self) -> dict:
        return asdict(self)


def perceptual_hash(image: "Image.Image | str", size: int = 8) -> str:
    """A dHash: 64 bits of "which way did the brightness step here".

    Resize to (size+1) x size greyscale and compare each pixel with its right
    neighbour. Rescaling and re-encoding barely move the answer, which is what
    byte hashes get wrong: the same hero shot at 800px and 1600px is one image
    to a content team and two files to sha256.
    """
    img = Image.open(image) if isinstance(image, str) else image
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = small.tobytes()               # mode "L", so one byte per pixel
    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            bits <<= 1
            bits |= int(pixels[offset + col] > pixels[offset + col + 1])
    return f"{bits:0{size * size // 4}x}"


def hamming(a: str, b: str) -> int:
    """Bits differing between two hashes -- 0 is identical, <=5 is the same
    shot re-encoded, and anything above ~10 is a different picture."""
    if len(a) != len(b):
        raise ValueError(f"hashes are different lengths: {len(a)} vs {len(b)}")
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _exif(img: "Image.Image") -> dict:
    """The handful of EXIF fields worth keeping, by their readable names."""
    try:
        raw = img.getexif()
    except Exception:                      # a truncated or odd EXIF block is
        return {}                          # not a reason to lose the file
    out: dict = {}
    for tag, value in (raw or {}).items():
        field = _EXIF_WANTED.get(_EXIF_NAMES.get(tag, ""))
        if not field or (field in out and out[field]):
            continue                       # DateTimeOriginal beats DateTime
        if field == "orientation":
            out[field] = int(value) if str(value).isdigit() else 0
        else:
            out[field] = str(value).strip().strip("\x00")
    return out


def read_metadata(path: str, category: str = "", product: str = "",
                  data: bytes | None = None) -> ImageMeta:
    """Read one image. `category`/`product` default to what the folder path
    says, so a fetched image and an uploaded one describe themselves the same
    way. Pass `data` to read bytes already in hand without a second open."""
    if not category or not product:
        folder_category, folder_product = _split_context(_infer_context(path))
        category = category or folder_category
        product = product or folder_product
    if data is None:
        with open(path, "rb") as fh:
            data = fh.read()
    with Image.open(path) as img:
        width, height = img.size
        meta = ImageMeta(
            path=path, name=os.path.basename(path), category=category,
            product=product, fmt=img.format or "", width=width, height=height,
            mode=img.mode, bytes=len(data),
            mime=MIME_TYPES.get(img.format or "", "application/octet-stream"),
            sha256=hashlib.sha256(data).hexdigest(),
            phash=perceptual_hash(img), **_exif(img))
    log.info("metadata_read", extra={"image": path, "w": meta.width,
                                     "h": meta.height, "phash": meta.phash})
    return meta
