"""Few-shot examples drawn from the labelled training images.

The training filenames already carry their tags, so the labelled set doubles as
a prompt-time teaching set: showing the model a handful of correctly tagged
images is far cheaper than fine-tuning and needs no extra annotation.

Kept in its own module so `pipeline` stays free of any dependency on the
evaluation layer; the label parsing it needs lives in `labels`.
"""
from __future__ import annotations

import os
import random

from .labels import load_labelled
from .vlm import Example, encode_image

DEFAULT_TRAIN_DIR = "data/train"


def load_examples(train_dir: str = DEFAULT_TRAIN_DIR,
                  limit: int | None = None,
                  exclude: str | None = None,
                  seed: int | None = None) -> list[Example]:
    """Encode labelled training images as (base64, media_type, tags) triples.

    `exclude` drops a single image path. Evaluation runs over the same labelled
    folder the examples come from, so without this an image would be shown its
    own answer and the reported score would be meaningless.

    `seed` shuffles before taking `limit`, which is what the self-consistency
    check uses to ask the same question with a different set of examples.
    Deterministic for a given seed, so a run can be repeated exactly.
    """
    pairs = load_labelled(train_dir)
    if exclude:
        target = os.path.abspath(exclude)
        pairs = [(p, t) for p, t in pairs if os.path.abspath(p) != target]
    if seed is not None:
        random.Random(seed).shuffle(pairs)
    if limit is not None:
        pairs = pairs[:limit]
    return [(*encode_image(path), tags) for path, tags in pairs]
