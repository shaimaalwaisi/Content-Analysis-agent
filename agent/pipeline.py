"""Batch-tag a folder of product images with the agent."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from .graph import _infer_context, _split_context, build_graph
from .logconf import get_logger
from .memory import TagMemory
from .taxonomy import highlight_tags
from .vlm import Example, VLMClient

if TYPE_CHECKING:            # typing only: the core must not depend at
    from evaluation.runstats import RunStats   # runtime on the layers that
    from tools import ResultStore, SearchTool  # measure it or extend it

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class TagResult:
    path: str
    category: str
    model: str
    tags: list[str]
    highlights: list[str] = field(default_factory=list)


def find_images(root: str) -> list[str]:
    """Recursively collect image files (handles the nested Category/Model
    folders in both the train and test sets)."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def run_folder(root: str, client: VLMClient, limit: int | None = None,
               on_item=None,
               examples: list[Example] | None = None,
               memory: TagMemory | None = None,
               workers: int = 1, search_tool: "SearchTool | None" = None,
               stats: "RunStats | None" = None,
               store: "ResultStore | None" = None,
               run_id: str | None = None) -> list[TagResult]:
    """Tag every image under `root`. `on_item(i, total, result)` is an optional
    progress callback (used by the CLI / Streamlit UI). `examples` are few-shot
    demonstrations prepended to every request (see fewshot.load_examples).
    `memory` reuses tags already computed for identical requests. `workers`
    tags images in parallel -- the work is network-bound, so threads help even
    though they share one interpreter. Results keep folder order regardless.
    `store` writes one durable row per image under `run_id`, which is what the
    results table in the UI reads."""
    app = build_graph(client, memory=memory, search_tool=search_tool,
                      stats=stats, store=store, run_id=run_id)
    paths = find_images(root)
    if limit:
        paths = paths[:limit]

    def tag_path(path: str) -> TagResult:
        ctx = _infer_context(path)
        started = time.perf_counter()
        if stats:
            stats.record_task()
        try:
            out = app.invoke({"image_path": path, "context": ctx or None,
                              "examples": examples})
            tags = out.get("tags", [])
        except Exception as exc:  # keep going on a single bad image
            tags = []
            if stats:
                stats.record_failure()
            log.error("image_failed", extra={"image": path,
                                             "error_type": type(exc).__name__,
                                             "error": str(exc)[:300]})
            print(f"  ! {os.path.basename(path)}: {exc}")
        else:
            log.info("image_tagged", extra={
                "image": path, "n_tags": len(tags),
                "ms": round((time.perf_counter() - started) * 1000),
                "cached": bool(out.get("cached"))})
        if stats:
            # Outside the else: a task that raised is a task that returned no
            # tags, and both must count against the success rate.
            stats.record_outcome(len(tags))
        category, model = _split_context(ctx)
        return TagResult(path=path, category=category, model=model, tags=tags,
                         highlights=highlight_tags(tags))

    started = time.perf_counter()
    if workers > 1:
        # Submit in order and read the futures in order, so parallelism never
        # reorders the output or the progress callback.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(tag_path, p) for p in paths]
            results = []
            for i, future in enumerate(futures, 1):
                res = future.result()
                results.append(res)
                if on_item:
                    on_item(i, len(paths), res)
    else:
        results = []
        for i, path in enumerate(paths, 1):
            res = tag_path(path)
            results.append(res)
            if on_item:
                on_item(i, len(paths), res)

    log.info("run_complete", extra={
        "images": len(results), "workers": workers,
        "seconds": round(time.perf_counter() - started, 2)})
    return results


def results_to_dicts(results: list[TagResult]) -> list[dict]:
    return [asdict(r) for r in results]
