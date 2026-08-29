"""Vision-language model clients.

A thin, swappable interface so the agent does not care which provider tags the
image. Three implementations ship:

* AnthropicVLM  - default, Claude vision (claude-sonnet-5 by default)
* OpenAIVLM     - any OpenAI-compatible vision endpoint: OpenAI itself, and
                  equally xAI (Grok), Groq, or a local Ollama, which all speak
                  the same wire format and differ only by base URL and key
* MockVLM       - no network / no key; deterministic, for tests & offline demos

Prompt wording lives in prompts.py; a client only sends the prompt with the
image and extracts the model's tag list. Validation against the taxonomy
happens later (graph.validate_tags).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
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


class VLMClient(Protocol):
    def predict_tags(self, image_b64: str, media_type: str,
                     context: str | None = None,
                     examples: list[Example] | None = None) -> list[str]:
        ...


@dataclass
class AnthropicVLM:
    """Claude vision. Requires ANTHROPIC_API_KEY and `pip install anthropic`."""

    model: str = "claude-sonnet-5"
    max_tokens: int = 300

    def __post_init__(self) -> None:
        from anthropic import Anthropic  # lazy: mock needs no dependency
        self._client = Anthropic()

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        instruction = build_tagging_prompt(context)
        messages = []
        for ex_b64, ex_media, ex_tags in examples or []:
            messages.append({"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": ex_media, "data": ex_b64}},
                {"type": "text", "text": instruction}]})
            messages.append({"role": "assistant",
                             "content": json.dumps(ex_tags)})
        messages.append({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": instruction}]})
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, messages=messages)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_tag_array(text)


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

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        instruction = build_tagging_prompt(context)
        messages = []
        for ex_b64, ex_media, ex_tags in examples or []:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": instruction},
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
        return parse_tag_array(resp.choices[0].message.content or "")


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


# Endpoints that speak the OpenAI wire format. Only the URL, the key variable
# and the default model differ -- the request body is identical, which is why
# one client class covers all of them. Model ids move quickly on these
# services; override with --model if a default has been retired.
OPENAI_COMPATIBLE = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY",
               "model": "gpt-4o"},
    "xai": {"base_url": "https://api.x.ai/v1", "key_env": "XAI_API_KEY",
            "model": "grok-2-vision-1212"},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY",
             "model": "meta-llama/llama-4-scout-17b-16e-instruct"},
    "ollama": {"base_url": "http://localhost:11434/v1",
               "key_env": "OLLAMA_API_KEY", "model": "llama3.2-vision",
               "key_fallback": "ollama"},
}

PROVIDERS = ["anthropic", *OPENAI_COMPATIBLE, "mock"]


def get_client(provider: str, model: str | None = None) -> VLMClient:
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicVLM(model=model or "claude-sonnet-5")
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


def encode_image(path: str, max_dim: int = MAX_IMAGE_DIM) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file.

    Images longer than `max_dim` on their longest side are downscaled and
    re-encoded as JPEG; smaller ones are sent untouched. Pass max_dim=0 to
    send the original bytes regardless.
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "webp": "image/webp",
                  "gif": "image/gif"}.get(ext, "image/jpeg")

    if max_dim:
        try:
            from PIL import Image
            with Image.open(path) as im:
                if max(im.size) > max_dim:
                    shrunk = im.convert("RGB")
                    shrunk.thumbnail((max_dim, max_dim), Image.LANCZOS)
                    buf = io.BytesIO()
                    shrunk.save(buf, format="JPEG", quality=JPEG_QUALITY)
                    data = buf.getvalue()
                    return base64.standard_b64encode(data).decode(), "image/jpeg"
        except Exception:
            pass  # Pillow missing or file unreadable: fall back to raw bytes
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), media_type
