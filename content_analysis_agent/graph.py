"""The tagging agent, built as a small LangGraph state machine.

Flow (one run per image):

    load_image -> recall -+-(hit)------------------------> END
                          |
                          +-(miss)-> tag_image -> validate_tags -> remember -> END

Each node is a small, testable function. This makes the design easy to read
and extend (e.g. drop an `enrich` node between tag and validate to look up
non-visual tags like 'awards'/'benchmark' via web search or MCP).

`recall`/`remember` are the agent's memory: with a TagMemory attached, an image
already tagged under identical conditions short-circuits straight to END
without calling the model.
"""
from __future__ import annotations

import os
import time
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from .logconf import get_logger
from .memory import TagMemory, make_key
from .retry import call_with_retry
from .runstats import RunStats
from .tools import SearchTool, tags_from_evidence
from .taxonomy import allowed_tags, normalise
from .vlm import Example, VLMClient, encode_image

log = get_logger(__name__)


class TagState(TypedDict, total=False):
    image_path: str
    context: Optional[str]          # e.g. "Category: Mobile, Model: XPERIA10MK5"
    examples: Optional[list[Example]]
    image_b64: str
    media_type: str
    raw_tags: list[str]             # what the model said
    evidence: list                  # search results backing non-visual tags
    enriched_tags: list[str]        # tags proposed by the enrich step
    tags: list[str]                 # validated, in-vocabulary tags
    cache_key: Optional[str]        # memory fingerprint for this request
    cached: bool                    # True when tags came from memory
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


def build_graph(client: VLMClient, memory: TagMemory | None = None,
                attempts: int = 3, search_tool: SearchTool | None = None,
                stats: RunStats | None = None):
    """Compile the agent for a given VLM client.

    Pass a TagMemory to reuse previously computed tags for identical requests.
    `attempts` bounds how often a transient provider failure is retried.
    Passing a `search_tool` inserts the enrich node, which looks the product up
    to justify non-visual tags (awards, benchmark, energy rating).
    Passing a `RunStats` collects the label-free workflow metrics.
    """
    _retry = (lambda _e: stats.record_retry()) if stats else None
    model_id = getattr(client, "model", type(client).__name__)

    def load_image(state: TagState) -> TagState:
        path = state["image_path"]
        b64, media = encode_image(path)
        ctx = state.get("context") or _infer_context(path) or None
        return {"image_b64": b64, "media_type": media, "context": ctx}

    def recall(state: TagState) -> TagState:
        """Look the request up in memory before spending a model call."""
        if memory is None:
            return {"cached": False}
        key = make_key(state["image_b64"], model_id, state.get("context"),
                       state.get("examples"),
                       extra="enrich" if search_tool else "")
        hit = memory.get(key)
        if hit is not None:
            log.info("memory_hit", extra={"image": state["image_path"],
                                          "n_tags": len(hit)})
            if stats:
                stats.record_cache_hit()
            return {"cache_key": key, "cached": True, "tags": hit}
        return {"cache_key": key, "cached": False}

    def tag_image(state: TagState) -> TagState:
        started = time.perf_counter()
        raw = call_with_retry(
            lambda: client.predict_tags(
                state["image_b64"], state["media_type"],
                context=state.get("context"),
                examples=state.get("examples")),
            attempts=attempts, on_retry=_retry)
        log.info("model_call", extra={
            "image": state["image_path"], "model": model_id,
            "ms": round((time.perf_counter() - started) * 1000),
            "n_raw_tags": len(raw),
            "n_examples": len(state.get("examples") or [])})
        if stats:
            stats.record_model_call((time.perf_counter() - started) * 1000,
                                    len(raw))
        return {"raw_tags": raw}

    def enrich(state: TagState) -> TagState:
        """Look the product up to justify tags the image cannot show.

        Only ever *proposes* tags: everything still passes through
        validate_tags, so the controlled vocabulary is enforced regardless of
        what a search returns.
        """
        context = state.get("context") or ""
        if not context:
            return {"enriched_tags": []}
        # Ask for the evidence the rules look for. A bare product query
        # returns generic marketing copy that mentions none of it.
        product = (context.replace("Category:", "")
                          .replace("Model:", "").replace(",", " ").strip())
        query = (f"Sony {product}: has it won any awards, what do benchmark "
                 f"or lab test results say, and what is its energy rating?")
        started = time.perf_counter()
        try:
            results = call_with_retry(lambda: search_tool.search(query.strip()),
                                      attempts=attempts, on_retry=_retry)
        except Exception as exc:
            # Enrichment is additive; a search failure must not lose the tags
            # the model already produced.
            log.warning("enrich_failed", extra={
                "image": state["image_path"],
                "error_type": type(exc).__name__, "error": str(exc)[:200]})
            return {"enriched_tags": []}
        if stats:
            stats.record_tool_call((time.perf_counter() - started) * 1000)
        proposed = tags_from_evidence(results)
        log.info("enriched", extra={
            "image": state["image_path"], "query": query.strip(),
            "results": len(results), "proposed": proposed,
            "ms": round((time.perf_counter() - started) * 1000)})
        return {"evidence": [r.__dict__ for r in results],
                "enriched_tags": proposed,
                "raw_tags": list(state.get("raw_tags", [])) + proposed}

    def validate_tags(state: TagState) -> TagState:
        vocab = allowed_tags()
        seen, clean = set(), []
        for t in state.get("raw_tags", []):
            n = normalise(t)
            if n in vocab and n not in seen:
                seen.add(n)
                clean.append(n)
        dropped = [t for t in state.get("raw_tags", [])
                   if normalise(t) not in vocab]
        if dropped:
            # Out-of-vocabulary predictions are the signal to watch in
            # production: a rising rate means prompt or taxonomy drift.
            log.warning("out_of_vocab_tags", extra={
                "image": state["image_path"], "dropped": dropped,
                "kept": clean})
        # Counted only on a real model answer: a memory hit replays an
        # already-validated list and would dilute the rate with zeros.
        if stats and not state.get("cached"):
            stats.record_dropped(len(dropped))
        return {"tags": clean}

    def remember(state: TagState) -> TagState:
        """Store the validated tags so an identical request skips the model."""
        if memory is not None and state.get("cache_key"):
            memory.put(state["cache_key"], state.get("tags", []), model_id)
        return {}

    g = StateGraph(TagState)
    g.add_node("load_image", load_image)
    g.add_node("recall", recall)
    g.add_node("tag_image", tag_image)
    g.add_node("validate_tags", validate_tags)
    g.add_node("remember", remember)
    g.set_entry_point("load_image")
    g.add_edge("load_image", "recall")
    # A memory hit already has its tags: route straight to the end.
    g.add_conditional_edges("recall", lambda s: END if s.get("cached")
                            else "tag_image", {END: END,
                                               "tag_image": "tag_image"})
    if search_tool is not None:
        g.add_node("enrich", enrich)
        g.add_edge("tag_image", "enrich")
        g.add_edge("enrich", "validate_tags")
    else:
        g.add_edge("tag_image", "validate_tags")
    g.add_edge("validate_tags", "remember")
    g.add_edge("remember", END)
    return g.compile()


def tag_one(app, image_path: str, context: str | None = None,
            examples: list[Example] | None = None) -> list[str]:
    """Run the compiled graph for a single image, return validated tags."""
    out = app.invoke({"image_path": image_path, "context": context,
                      "examples": examples})
    return out.get("tags", [])
