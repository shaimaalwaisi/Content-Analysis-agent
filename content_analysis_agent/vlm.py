"""Vision-language model clients.

A thin, swappable interface so the agent does not care which provider tags the
image. Three implementations ship:

* AnthropicVLM  - default, Claude vision (claude-sonnet-5 by default)
* OpenAIVLM     - alternative, a GPT-4o-class vision model
* MockVLM       - no network / no key; deterministic, for tests & offline demos

Prompt wording lives in prompts.py; a client only sends the prompt with the
image and extracts the model's tag list. Validation against the taxonomy
happens later (graph.validate_tags).
"""
from __future__ import annotations

import base64
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
    """GPT-4o-class vision. Requires OPENAI_API_KEY and `pip install openai`."""

    model: str = "gpt-4o"
    max_tokens: int = 300

    def __post_init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI()

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


def get_client(provider: str, model: str | None = None) -> VLMClient:
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicVLM(model=model or "claude-sonnet-5")
    if provider == "openai":
        return OpenAIVLM(model=model or "gpt-4o")
    if provider == "mock":
        return MockVLM()
    raise ValueError(f"Unknown provider: {provider!r}")


def encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "webp": "image/webp",
                  "gif": "image/gif"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), media_type
