"""Vision-language model clients.

A thin, swappable interface so the agent does not care which provider tags the
image. Three implementations ship:

* AnthropicVLM  - default, Claude vision (claude-haiku-4-5 by default)
* OpenAIVLM     - any OpenAI-compatible vision endpoint: OpenAI itself, and
                  equally xAI (Grok), Groq, or a local Ollama, which all speak
                  the same wire format and differ only by base URL and key
* MockVLM       - no network / no key; deterministic, for tests & offline demos

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

from .prompts import build_tagging_prompt

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

    reasons: dict[str, str] = {}
    for line in (text or "").splitlines():
        if line.lstrip().startswith("["):      # the answer array, not a reason
            continue
        match = _REASON_LINE.match(line)
        if not match:
            continue
        tag = " ".join(match.group("tag").lower().split())
        why = clean(match.group("why"))
        reasons.setdefault(tag, why)
        nested = _REASON_LINE.match(why)
        if nested:
            inner = " ".join(nested.group("tag").lower().split())
            if inner in vocabulary:
                reasons.setdefault(inner, clean(nested.group("why")))
    return reasons


@dataclass
class Prediction:
    """One model answer: the tags, and why it says it chose them."""

    tags: list[str]
    reasons: dict[str, str] = field(default_factory=dict)


class VLMClient(Protocol):
    def predict_tags(self, image_b64: str, media_type: str,
                     context: str | None = None,
                     examples: list[Example] | None = None) -> list[str]:
        ...


@dataclass
class AnthropicVLM:
    """Claude vision. Requires ANTHROPIC_API_KEY and `pip install anthropic`."""

    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 300

    def __post_init__(self) -> None:
        from anthropic import Anthropic  # lazy: mock needs no dependency
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
        return Prediction(parse_tag_array(text), parse_reasons(text))

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        return self.predict(image_b64, media_type, context, examples,
                            reasons=False).tags


@dataclass
class OpenAIVLM:
    """Any OpenAI-compatible vision endpoint. Requires `pip install openai`.

    `base_url` and `api_key_env` are all that separate the providers: leave
    them unset for OpenAI itself, or point them at xAI, Groq, or a local
    server. The request body is identical in every case.
    """

    model: str = "gpt-4o"
    max_tokens: int = 300
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_key_fallback: str | None = None   # local servers accept any string

    def __post_init__(self) -> None:
        from openai import OpenAI
        key = os.getenv(self.api_key_env) or self.api_key_fallback
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Export it, or add it to a "
                f".env file in the repo root.")
        self._client = OpenAI(api_key=key, base_url=self.base_url)

    def predict(self, image_b64, media_type, context=None, examples=None,
                reasons=True, feedback=None):
        instruction = build_tagging_prompt(context, reasons=reasons,
                                           feedback=feedback)
        example_instruction = build_tagging_prompt(context)
        messages = []
        for ex_b64, ex_media, ex_tags in examples or []:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": example_instruction},
                {"type": "image_url", "image_url": {
                    "url": f"data:{ex_media};base64,{ex_b64}"}}]})
            messages.append({"role": "assistant",
                             "content": json.dumps(ex_tags)})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {
                "url": f"data:{media_type};base64,{image_b64}"}}]})
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, messages=messages)
        text = resp.choices[0].message.content or ""
        return Prediction(parse_tag_array(text), parse_reasons(text))

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        return self.predict(image_b64, media_type, context, examples,
                            reasons=False).tags


@dataclass
class MockVLM:
    """Offline stand-in. Category-aware guesses so the pipeline runs with no
    API key. Good for wiring tests/demos, NOT for measuring real accuracy."""

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        ctx = (context or "").lower()
        if "mobile" in ctx:
            return ["physical design", "side angle"]
        if "tv" in ctx:
            return ["physical design", "front angle"]
        if "sound" in ctx or "headphone" in ctx or "speaker" in ctx:
            return ["physical design", "product summary"]
        return ["physical design"]

    def predict(self, image_b64, media_type, context=None, examples=None,
                reasons=True, feedback=None):
        tags = self.predict_tags(image_b64, media_type, context, examples)
        return Prediction(tags, {t: "mock client: no image was inspected"
                                 for t in tags} if reasons else {})


# Endpoints that speak the OpenAI wire format. Only the URL, the key variable
# and the default model differ -- the request body is identical, which is why
# one client class covers all of them. Model ids move quickly on these
# services; override with --model if a default has been retired.
OPENAI_COMPATIBLE = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY",
               "model": "gpt-4o"},
    "xai": {"base_url": "https://api.x.ai/v1", "key_env": "XAI_API_KEY",
            "model": "grok-2-vision-1212"},
    # Groq's catalogue is mostly text-only; qwen3.8-27b is the vision model
    # verified to accept image content on a free account.
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY", "model": "qwen/qwen3.8-27b"},
    "ollama": {"base_url": "http://localhost:11434/v1",
               "key_env": "OLLAMA_API_KEY", "model": "llama3.2-vision",
               "key_fallback": "ollama"},
}

PROVIDERS = ["anthropic", *OPENAI_COMPATIBLE, "mock"]


def get_client(provider: str, model: str | None = None) -> VLMClient:
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicVLM(model=model or "claude-haiku-4-5-20251001")
    if provider in OPENAI_COMPATIBLE:
        cfg = OPENAI_COMPATIBLE[provider]
        return OpenAIVLM(model=model or cfg["model"],
                         base_url=cfg["base_url"], api_key_env=cfg["key_env"],
                         api_key_fallback=cfg.get("key_fallback"))
    if provider == "mock":
        return MockVLM()
    raise ValueError(f"Unknown provider: {provider!r}. "
                     f"Choose one of: {', '.join(PROVIDERS)}")


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
