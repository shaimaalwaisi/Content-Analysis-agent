"""Vision-language model clients.

A thin interface, so the agent does not care what is behind it. One
implementation ships: `AnthropicVLM`, Claude vision on
`claude-haiku-4-5-20251001`.

One provider is a deliberate choice, not a limitation of the interface: the
prompt is tuned for one model, the cost table prices one model, and a second
vendor would double both without answering a question the brief asks. The
`VLMClient` protocol below is still the seam -- adding a provider means adding
a class, not editing the agent, and the test suite proves it by passing its
own stub through the same protocol.

Prompt wording lives in prompts.py; a client only sends the prompt with the
image and extracts the model's answer. Validation against the taxonomy happens
later, inside the graph's analyze_image node.

Two entry points, and the second is the one the agent uses:

* predict_tags(...) -> list[str]        the plain answer
* predict(...)      -> Prediction       tags plus the model's reason for each,
                                        and an optional `feedback` string that
                                        tells it what a previous attempt on the
                                        same image got wrong
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .prompts import DETAIL_FIELDS, build_tagging_prompt

# One example (image_b64, media_type, tags) for optional few-shot prompting.
Example = tuple[str, str, list[str]]


def parse_tag_array(text: str) -> list[str]:
    """Pull a JSON array of strings out of a model response, robustly."""
    if not text:
        return []
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(t) for t in data if isinstance(t, (str, int, float))]


# A reason line: "side angle - the phone is shot from its edge". Models write
# the separator as a hyphen, an en dash or a colon, and often bullet the line,
# so all of those are accepted; anything else is simply not a reason.
_REASON_LINE = re.compile(
    r"^\s*(?:[-*\u2022]\s*)?(?:\d+[.)]\s*)?"
    r"(?P<tag>[A-Za-z][A-Za-z &\']{1,30}?)\s*[-\u2013:]\s+(?P<why>\S.*)$")


def parse_reasons(text: str) -> dict[str, str]:
    """Pull `tag -> why` out of the model's step-1 lines.

    Free text, so this is best-effort by design: a missing or malformed reason
    costs an explanation, never a tag. Reasons for tags the model did not
    actually answer with are discarded by the caller.

    Models often write a General and its Specific on one line -- "feature
    graphics: camera - a ZEISS lens is called out" -- so a nested tag is read
    as well, but only when it is a real tag: without that check, any reason
    containing a dash would invent one.
    """
    from .taxonomy import allowed_tags       # cheap, and keeps the vocabulary
    vocabulary = allowed_tags()              # in one place

    def clean(value: str) -> str:
        return value.strip().rstrip(".")[:200]

    labels = {f.lower() for f in DETAIL_FIELDS}
    reasons: dict[str, str] = {}
    for line in (text or "").splitlines():
        if line.lstrip().startswith("["):      # the answer array, not a reason
            continue
        match = _REASON_LINE.match(line)
        if not match:
            continue
        tag = " ".join(match.group("tag").lower().split())
        if tag in labels:                      # "Category: Mobile" is a fact
            continue                           # about the product, not a tag
        why = clean(match.group("why"))
        reasons.setdefault(tag, why)
        nested = _REASON_LINE.match(why)
        if nested:
            inner = " ".join(nested.group("tag").lower().split())
            if inner in vocabulary:
                reasons.setdefault(inner, clean(nested.group("why")))
    return reasons


# "Category: Mobile", "Specs: 5000mAh, ZEISS". Case-insensitive, because a
# model that is asked for "Model:" will occasionally write "MODEL:".
_DETAIL_LINE = re.compile(
    r"^\s*\**\s*(?P<key>" + "|".join(DETAIL_FIELDS) + r")\s*\**\s*:\s*"
    r"(?P<value>\S.*)$", re.IGNORECASE)

# What the model writes when it has nothing to report; storing it verbatim
# would put the word "unknown" in a table cell, where a gap reads better.
_NOTHING = {"unknown", "none", "n/a", "na", "not shown", "not visible", "-"}


def parse_details(text: str) -> dict[str, str]:
    """Pull the labelled facts -- category, model, description, specs -- out
    of a reasoned answer. Missing or empty fields are simply absent."""
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        match = _DETAIL_LINE.match(line)
        if not match:
            continue
        value = match.group("value").strip().strip("*").strip()
        if value.lower().rstrip(".") in _NOTHING:
            continue
        out.setdefault(match.group("key").lower(), value[:300])
    return out


@dataclass
class Prediction:
    """One model answer: the tags, why it chose them, and what it read.

    The details are what a results table needs and the tag vocabulary cannot
    say -- which product this is, and what the image states about it.
    """

    tags: list[str]
    reasons: dict[str, str] = field(default_factory=dict)
    category: str = ""
    product: str = ""
    description: str = ""
    specs: str = ""
    # What the call was billed for. Zero when the caller reports no usage
    # (a client implementing only predict_tags, say), which is why cost per
    # task is reported as unpriced rather than as free.
    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_text(cls, text: str) -> "Prediction":
        """Read a whole reasoned answer: tags, reasons and details."""
        details = parse_details(text)
        return cls(parse_tag_array(text), parse_reasons(text),
                   category=details.get("category", ""),
                   product=details.get("model", ""),
                   description=details.get("description", ""),
                   specs=details.get("specs", ""))


def _with_usage(pred: "Prediction", resp) -> "Prediction":
    """Copy the response's token counts onto the prediction, if it has any.

    Read defensively: cost per task is a nice number to have, and losing a
    whole tagging run because a response object was shaped unexpectedly would
    be a poor trade.
    """
    usage = getattr(resp, "usage", None)
    if usage is not None:
        pred.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        pred.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return pred


class VLMClient(Protocol):
    def predict_tags(self, image_b64: str, media_type: str,
                     context: str | None = None,
                     examples: list[Example] | None = None) -> list[str]:
        ...


@dataclass
class AnthropicVLM:
    """Claude vision. Requires ANTHROPIC_API_KEY and `pip install anthropic`."""

    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 500      # tags, a reason each, and four detail lines

    def __post_init__(self) -> None:
        from anthropic import Anthropic  # lazy: only a real run needs it
        self._client = Anthropic()

    def predict(self, image_b64, media_type, context=None, examples=None,
                reasons=True, feedback=None):
        instruction = build_tagging_prompt(context, reasons=reasons,
                                           feedback=feedback)
        # Few-shot turns show the answer only: the vocabulary is what an
        # example teaches, and inventing a reason for someone else's label
        # would teach the model to invent them too.
        example_instruction = build_tagging_prompt(context)
        messages = []
        for ex_b64, ex_media, ex_tags in examples or []:
            messages.append({"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": ex_media, "data": ex_b64}},
                {"type": "text", "text": example_instruction}]})
            messages.append({"role": "assistant",
                             "content": json.dumps(ex_tags)})
        messages.append({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": instruction}]})
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, messages=messages)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _with_usage(Prediction.from_text(text), resp)

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        return self.predict(image_b64, media_type, context, examples,
                            reasons=False).tags


PROVIDERS = ["anthropic"]

# The one real model. Pinned here rather than spread across the CLI and the
# UI, so "which model answered" has a single answer -- and it is the id the
# cost table in `evaluation.runstats` prices against.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def get_client(provider: str = "anthropic",
               model: str | None = None) -> VLMClient:
    """Claude, and nothing else.

    One provider is deliberate: one prompt tuned for one model, one price card
    to keep honest, one thing to explain. The factory stays because the seam
    is worth keeping -- adding a provider means adding a class here, not
    editing the agent -- and because it is the single place the model id is
    chosen.

    Every run now needs ANTHROPIC_API_KEY. The test suite does not: it passes
    its own stub client straight to `build_graph`, so it still runs offline.
    """
    provider = provider.lower()
    if provider != "anthropic":
        raise ValueError(f"Unknown provider: {provider!r}. "
                         f"This project calls Claude only.")
    return AnthropicVLM(model=model or DEFAULT_MODEL)


# Product shots carry far more pixels than a tagging model needs, and the
# payload is billed and transferred per byte, so shrink before upload.
MAX_IMAGE_DIM = 1024   # px on the longest side
JPEG_QUALITY = 85


# Magic bytes, because a file's extension is not evidence of its format: the
# training data contains a PNG named .jpg, and providers validate the declared
# media type against the actual bytes and reject the mismatch.
_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_media_type(data: bytes, fallback: str = "image/jpeg") -> str:
    """Media type from the file's own bytes, not its name."""
    for signature, media_type in _MAGIC:
        if data.startswith(signature):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def encode_image(path: str, max_dim: int = MAX_IMAGE_DIM) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file.

    Images longer than `max_dim` on their longest side are downscaled and
    re-encoded as JPEG; smaller ones are sent untouched. Pass max_dim=0 to
    send the original bytes regardless. The media type is detected from the
    file's content rather than its extension.
    """
    with open(path, "rb") as f:
        data = f.read()
    media_type = sniff_media_type(data)

    if max_dim:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                if max(im.size) > max_dim:
                    shrunk = im.convert("RGB")
                    shrunk.thumbnail((max_dim, max_dim), Image.LANCZOS)
                    buf = io.BytesIO()
                    shrunk.save(buf, format="JPEG", quality=JPEG_QUALITY)
                    return (base64.standard_b64encode(buf.getvalue()).decode(),
                            "image/jpeg")
        except Exception:
            pass  # Pillow missing or file unreadable: send the raw bytes
    return base64.standard_b64encode(data).decode(), media_type
