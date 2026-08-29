"""The tagging agent, built as a small LangGraph state machine.

Flow (one run per image):

    load_image  ->  tag_image  ->  validate_tags  ->  END

Each node is a small, testable function. This makes the design easy to read
and extend (e.g. drop an `enrich` node between tag and validate to look up
non-visual tags like 'awards'/'benchmark' via web search or MCP).
"""
from __future__ import annotations

import os
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from .taxonomy import allowed_tags, normalise
from .vlm import Example, VLMClient, encode_image


class TagState(TypedDict, total=False):
    image_path: str
    context: Optional[str]          # e.g. "Category: Mobile, Model: XPERIA10MK5"
    examples: Optional[list[Example]]
    image_b64: str
    media_type: str
    raw_tags: list[str]             # what the model said
    tags: list[str]                 # validated, in-vocabulary tags
    error: Optional[str]


def _infer_context(image_path: str) -> str:
    """Best-effort product context from the folder path, e.g.
    .../Mobile/XPERIA10MK5/foo.jpg -> 'Category: Mobile, Model: XPERIA10MK5'.
    Test images are not in the metadata sheet, so the path is our context."""
    parts = [p for p in os.path.normpath(image_path).split(os.sep) if p]
    category = model = None
    known = {"mobile", "tv", "video & sound", "headphone", "speaker"}
    for i, p in enumerate(parts[:-1]):  # skip the filename itself
        if p.lower() in known:
            category = p
            if i + 1 < len(parts) - 1:
                model = parts[i + 1]
    bits = []
    if category:
        bits.append(f"Category: {category}")
    if model:
        bits.append(f"Model: {model}")
    return ", ".join(bits)


def build_graph(client: VLMClient):
    """Compile the agent for a given VLM client."""

    def load_image(state: TagState) -> TagState:
        path = state["image_path"]
        b64, media = encode_image(path)
        ctx = state.get("context") or _infer_context(path) or None
        return {"image_b64": b64, "media_type": media, "context": ctx}

    def tag_image(state: TagState) -> TagState:
        raw = client.predict_tags(
            state["image_b64"], state["media_type"],
            context=state.get("context"), examples=state.get("examples"))
        return {"raw_tags": raw}

    def validate_tags(state: TagState) -> TagState:
        vocab = allowed_tags()
        seen, clean = set(), []
        for t in state.get("raw_tags", []):
            n = normalise(t)
            if n in vocab and n not in seen:
                seen.add(n)
                clean.append(n)
        return {"tags": clean}

    g = StateGraph(TagState)
    g.add_node("load_image", load_image)
    g.add_node("tag_image", tag_image)
    g.add_node("validate_tags", validate_tags)
    g.set_entry_point("load_image")
    g.add_edge("load_image", "tag_image")
    g.add_edge("tag_image", "validate_tags")
    g.add_edge("validate_tags", END)
    return g.compile()


def tag_one(app, image_path: str, context: str | None = None,
            examples: list[Example] | None = None) -> list[str]:
    """Run the compiled graph for a single image, return validated tags."""
    out = app.invoke({"image_path": image_path, "context": context,
                      "examples": examples})
    return out.get("tags", [])
