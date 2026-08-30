"""The tagging agent, built as a small LangGraph state machine.

Three nodes, one branch and one loop (per image):

    prepare -+-(memory hit)------------------------------+-> persist -> END
             |                                           |
             +-(miss)-> analyze_image -+-(good answer)---+
                            ^          |
                            +-(weak)---+   at most `attempts` passes

* prepare       reads the image, infers the product context from its path and
                asks memory whether this exact request was answered before.
* analyze_image is the reason-act-check step: the model proposes tags *with a
                reason for each*, an optional search tool adds the tags an
                image cannot show, and the controlled vocabulary decides what
                survives. If too little survives, the node loops once with the
                rejected tags fed back into the prompt.
* persist       writes the answer: the memory row that lets an identical
                request skip the model, and the durable results row the UI
                reads. A memory hit routes through here too, so ten images
                always produce ten rows.

The nodes stay small and testable, and the shape no longer changes with
configuration: enrichment is a step inside analyze_image rather than a fourth
box that appears only when a search tool is wired in.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .enrichment import tags_from_evidence
from .logconf import get_logger
from .memory import TagMemory, make_key
from .prompts import build_feedback
from .retry import call_with_retry
from .taxonomy import allowed_tags, highlight_tags, normalise
from .vlm import Example, VLMClient, encode_image

if TYPE_CHECKING:            # typing only: the core must not depend at
    from evaluation.runstats import RunStats   # runtime on the layers that
    from tools import ResultStore, SearchTool  # measure it or extend it

log = get_logger(__name__)

# How many times analyze_image may look at one image. 2 = one re-prompt after
# a weak answer, which is the whole reasoning loop; 1 disables it.
DEFAULT_ATTEMPTS = 2


class TagState(TypedDict, total=False):
    image_path: str
    context: Optional[str]          # e.g. "Category: Mobile, Model: XPERIA10MK5"
    examples: Optional[list[Example]]
    run_id: Optional[str]           # the batch this image belongs to
    image_b64: str
    media_type: str
    raw_tags: list[str]             # what the model said
    rationale: dict                 # tag -> the model's reason for it
    description: str                # one line: what the image shows
    specs: str                      # figures the model could read in the image
    seen_category: str              # the category the model recognised
    seen_product: str               # the model name it could read
    evidence: list                  # search results backing non-visual tags
    dropped: list[str]              # tags rejected by the vocabulary
    feedback: Optional[str]         # what to tell the model on a second pass
    attempt: int                    # passes through analyze_image so far
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


def _split_context(context: str | None) -> tuple[str, str]:
    """'Category: TV, Model: XR-65A95K' -> ('TV', 'XR-65A95K')."""
    category = product = ""
    for part in (context or "").split(","):
        part = part.strip()
        if part.lower().startswith("category:"):
            category = part.split(":", 1)[1].strip()
        elif part.lower().startswith("model:"):
            product = part.split(":", 1)[1].strip()
    return category, product


def build_graph(client: VLMClient, memory: TagMemory | None = None,
                attempts: int = 3, search_tool: "SearchTool | None" = None,
                stats: "RunStats | None" = None,
                store: "ResultStore | None" = None,
                run_id: str | None = None,
                passes: int = DEFAULT_ATTEMPTS):
    """Compile the agent for a given VLM client.

    Pass a TagMemory to reuse previously computed tags for identical requests.
    `attempts` bounds how often a transient provider failure is retried.
    Passing a `search_tool` turns on enrichment, which looks the product up to
    justify non-visual tags (awards, benchmark, energy rating).
    Passing a `RunStats` collects the label-free workflow metrics.
    Passing a `ResultStore` writes one durable row per image, tagged with
    `run_id` (state may override it per image).
    `passes` bounds the reasoning loop: 2 allows one re-prompt after a weak
    answer, 1 turns the loop off.
    """
    _retry = (lambda _e: stats.record_retry()) if stats else None
    model_id = getattr(client, "model", type(client).__name__)

    def prepare(state: TagState) -> TagState:
        """Read the image, work out the context, ask memory."""
        path = state["image_path"]
        started = time.perf_counter()
        b64, media = encode_image(path)
        if stats:
            stats.record_action("encode",
                                (time.perf_counter() - started) * 1000)
        ctx = state.get("context") or _infer_context(path) or None
        out: TagState = {"image_b64": b64, "media_type": media, "context": ctx,
                         "attempt": 0, "cached": False}
        if memory is None:
            return out
        key = make_key(b64, model_id, ctx, state.get("examples"),
                       extra="enrich" if search_tool else "")
        out["cache_key"] = key
        hit = memory.get_record(key)
        if hit is not None:
            log.info("memory_hit", extra={"image": path,
                                          "n_tags": len(hit["tags"])})
            if stats:
                stats.record_cache_hit()
            details = hit.get("details") or {}
            out.update({"cached": True, "tags": hit["tags"],
                        "rationale": hit.get("rationale") or {},
                        "description": details.get("description", ""),
                        "specs": details.get("specs", ""),
                        "seen_category": details.get("category", ""),
                        "seen_product": details.get("product", "")})
        return out

    def _call_model(state: TagState):
        """One model call. Clients that support it return reasons and the
        details a results table needs; a plain client (a test stub, or
        anything implementing only the protocol's predict_tags) still works
        and simply has nothing to explain."""
        from .vlm import Prediction
        predict = getattr(client, "predict", None)
        if predict is None:
            tags = client.predict_tags(
                state["image_b64"], state["media_type"],
                context=state.get("context"), examples=state.get("examples"))
            return Prediction(list(tags))
        return predict(state["image_b64"], state["media_type"],
                       context=state.get("context"),
                       examples=state.get("examples"),
                       feedback=state.get("feedback"))

    def _enrich(state: TagState) -> tuple[list[str], list]:
        """Look the product up to justify tags the image cannot show.

        Only ever *proposes* tags: everything still passes through validation,
        so the controlled vocabulary is enforced regardless of what a search
        returns. Enrichment is additive -- a search failure must not lose the
        tags the model already produced.
        """
        context = state.get("context") or ""
        if not context:
            return [], []
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
            log.warning("enrich_failed", extra={
                "image": state["image_path"],
                "error_type": type(exc).__name__, "error": str(exc)[:200]})
            return [], []
        if stats:
            stats.record_tool_call((time.perf_counter() - started) * 1000)
        proposed = tags_from_evidence(results)
        log.info("enriched", extra={
            "image": state["image_path"], "query": query.strip(),
            "results": len(results), "proposed": proposed,
            "ms": round((time.perf_counter() - started) * 1000)})
        return proposed, [r.__dict__ for r in results]

    def _validate(raw: list[str]) -> tuple[list[str], list[str]]:
        """Split a raw answer into (kept, dropped) against the vocabulary."""
        vocab = allowed_tags()
        seen, kept, dropped = set(), [], []
        for tag in raw:
            norm = normalise(tag)
            if norm in vocab and norm not in seen:
                seen.add(norm)
                kept.append(norm)
            elif norm not in vocab:
                dropped.append(tag)
        return kept, dropped

    def analyze_image(state: TagState) -> TagState:
        """Reason, act, check: ask the model, enrich, enforce the vocabulary."""
        started = time.perf_counter()
        pred = call_with_retry(lambda: _call_model(state),
                               attempts=attempts, on_retry=_retry)
        raw, reasons = list(pred.tags), dict(pred.reasons)
        log.info("model_call", extra={
            "image": state["image_path"], "model": model_id,
            "ms": round((time.perf_counter() - started) * 1000),
            "n_raw_tags": len(raw), "attempt": state.get("attempt", 0) + 1,
            "n_examples": len(state.get("examples") or [])})
        if stats:
            stats.record_model_call((time.perf_counter() - started) * 1000,
                                    pred.input_tokens, pred.output_tokens)

        evidence = []
        if search_tool is not None:
            proposed, evidence = _enrich(state)
            raw = raw + proposed

        kept, dropped = _validate(raw)
        if dropped:
            # Out-of-vocabulary predictions are the signal to watch in
            # production: a rising rate means prompt or taxonomy drift. It is
            # logged rather than counted -- the run metrics are the three in
            # `evaluation`, and this is an observability signal, not a fourth.
            log.warning("out_of_vocab_tags", extra={
                "image": state["image_path"], "dropped": dropped,
                "kept": kept})
        return {"raw_tags": raw, "tags": kept, "dropped": dropped,
                "evidence": evidence, "description": pred.description,
                "specs": pred.specs, "seen_category": pred.category,
                "seen_product": pred.product,
                # Only reasons for tags that survived: an explanation for a
                # rejected tag is an explanation of something that never
                # happened.
                "rationale": {t: reasons[t] for t in kept if t in reasons},
                "attempt": state.get("attempt", 0) + 1,
                "feedback": build_feedback(dropped, kept)}

    def _looks_weak(state: TagState) -> bool:
        """Is this answer worth a second look?

        Two symptoms, both cheap to measure and both meaning the model did not
        work inside the vocabulary: nothing survived validation, or more tags
        were rejected than kept.
        """
        kept, dropped = state.get("tags", []), state.get("dropped", [])
        return not kept or len(dropped) > len(kept)

    def route_after_analyze(state: TagState) -> str:
        if state.get("attempt", 1) < passes and _looks_weak(state):
            log.info("reasoning_retry", extra={
                "image": state["image_path"], "attempt": state.get("attempt"),
                "kept": state.get("tags", []),
                "dropped": state.get("dropped", [])})
            return "analyze_image"
        return "persist"

    def persist(state: TagState) -> TagState:
        """Write the answer: the memory row, and the durable result row."""
        tags = state.get("tags", [])
        if memory is not None and state.get("cache_key") \
                and not state.get("cached"):
            memory.put(state["cache_key"], tags, model_id,
                       rationale=state.get("rationale", {}),
                       details={"description": state.get("description", ""),
                                "specs": state.get("specs", ""),
                                "category": state.get("seen_category", ""),
                                "product": state.get("seen_product", "")})
        if store is not None:
            from tools import Tagging      # the core never imports tools at
            batch = state.get("run_id") or run_id  # import time
            # What we were told beats what the model reckons: the folder path
            # (or the person at the keyboard) knows the product, while the
            # model is reading a name off a photograph.
            category, product = _split_context(state.get("context"))
            store.put(Tagging(
                run_id=batch or "adhoc", image_path=state["image_path"],
                tags=tags, highlights=highlight_tags(tags),
                rationale=state.get("rationale", {}),
                category=category or state.get("seen_category", ""),
                product=product or state.get("seen_product", ""),
                description=state.get("description", ""),
                specs=state.get("specs", ""), model=model_id,
                attempts=state.get("attempt", 0),
                cached=bool(state.get("cached"))))
        return {}

    g = StateGraph(TagState)
    g.add_node("prepare", prepare)
    g.add_node("analyze_image", analyze_image)
    g.add_node("persist", persist)
    g.set_entry_point("prepare")
    # A memory hit already has its tags, but still needs a results row.
    g.add_conditional_edges("prepare",
                            lambda s: "persist" if s.get("cached")
                            else "analyze_image",
                            {"persist": "persist",
                             "analyze_image": "analyze_image"})
    # The reasoning loop: a weak answer goes back to the same node, once.
    g.add_conditional_edges("analyze_image", route_after_analyze,
                            {"analyze_image": "analyze_image",
                             "persist": "persist"})
    g.add_edge("persist", END)
    return g.compile()


def tag_one(app, image_path: str, context: str | None = None,
            examples: list[Example] | None = None,
            run_id: str | None = None) -> list[str]:
    """Run the compiled graph for a single image, return validated tags."""
    out = app.invoke({"image_path": image_path, "context": context,
                      "examples": examples, "run_id": run_id})
    return out.get("tags", [])
