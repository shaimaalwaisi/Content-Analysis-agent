"""Ground-truth tags encoded in training filenames.

Training images are named as a literal Python list::

    ['physical design', 'side angle', 'top'].jpg

so every training image carries its own labels. That makes them useful in two
unrelated places -- as few-shot examples and as an evaluation set -- which is
why the parsing lives in the core package rather than in `evaluation`:
dependencies point inward, and `fewshot` must not have to import the
evaluation layer to read a filename.
"""
from __future__ import annotations

import ast
import os
import re

from .pipeline import find_images
from .taxonomy import normalise


def parse_tags_from_filename(name: str) -> list[str]:
    """Extract the bracketed tag list from a filename. Returns [] if none."""
    base = os.path.basename(name)
    match = re.search(r"\[.*\]", base, re.DOTALL)
    if not match:
        return []
    try:
        tags = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(tags, (list, tuple)):
        return []
    return [normalise(t) for t in tags]


def load_labelled(root: str) -> list[tuple[str, list[str]]]:
    """(image_path, ground_truth_tags) for every labelled image under root."""
    out = []
    for path in find_images(root):
        tags = parse_tags_from_filename(path)
        if tags:
            out.append((path, tags))
    return out
