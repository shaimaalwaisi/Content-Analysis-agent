"""Batch-tag a folder of product images with the agent."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .graph import _infer_context, build_graph
from .vlm import Example, VLMClient

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class TagResult:
    path: str
    category: str
    model: str
    tags: list[str]


def find_images(root: str) -> list[str]:
    """Recursively collect image files (handles the nested Category/Model
    folders in both the train and test sets)."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _split_context(ctx: str) -> tuple[str, str]:
    category = model = ""
    for part in ctx.split(","):
        part = part.strip()
        if part.lower().startswith("category:"):
            category = part.split(":", 1)[1].strip()
        elif part.lower().startswith("model:"):
            model = part.split(":", 1)[1].strip()
    return category, model


def run_folder(root: str, client: VLMClient, limit: int | None = None,
               on_item=None,
               examples: list[Example] | None = None) -> list[TagResult]:
    """Tag every image under `root`. `on_item(i, total, result)` is an optional
    progress callback (used by the CLI / Streamlit UI). `examples` are few-shot
    demonstrations prepended to every request (see fewshot.load_examples)."""
    app = build_graph(client)
    paths = find_images(root)
    if limit:
        paths = paths[:limit]

    results: list[TagResult] = []
    for i, path in enumerate(paths, 1):
        ctx = _infer_context(path)
        try:
            out = app.invoke({"image_path": path, "context": ctx or None,
                              "examples": examples})
            tags = out.get("tags", [])
        except Exception as exc:  # keep going on a single bad image
            tags = []
            print(f"  ! {os.path.basename(path)}: {exc}")
        category, model = _split_context(ctx)
        res = TagResult(path=path, category=category, model=model, tags=tags)
        results.append(res)
        if on_item:
            on_item(i, len(paths), res)
    return results


def results_to_dicts(results: list[TagResult]) -> list[dict]:
    return [asdict(r) for r in results]
