"""The textual half of the prompt, so wording can be tuned in one place.

There is currently a single instruction: the image-tagging prompt, kept as
named text blocks plus a small builder that injects the controlled vocabulary
and the per-image product context.

Note that this is not the *whole* prompt. Few-shot examples are prompt content
too, but a vision API carries them as alternating user/assistant message turns
rather than as text, so they are prepared in `fewshot.py` (which loads and
encodes the example images) and assembled by the clients in `vlm.py`. The split
keeps this module pure text with one dependency; merging the two would pull
file I/O and the evaluation module into the prompt layer.
"""
from __future__ import annotations

from .taxonomy import taxonomy_prompt

# What the model is asked to do.
INSTRUCTION = (
    "You label product images for a retail marketing team. "
    "Choose ALL tags from the controlled vocabulary below that apply to "
    "the image. Tags come in two levels: a General category and its "
    "Specifics. Include the General category and every Specific you can "
    "clearly see. Most images have between 1 and 4 tags. Only use tags "
    "from this vocabulary; never invent new ones."
)

# How the answer must be formatted (kept strict so parsing is reliable).
OUTPUT_FORMAT = (
    'Respond with ONLY a JSON array of tag strings, e.g. '
    '["physical design", "side angle", "top"]. No other text.\n'
    'Each array element must be exactly one tag, copied verbatim from the '
    'vocabulary. Never join a General category to its Specific in one string: '
    'write ["feature graphics", "camera"], not ["feature graphics: camera"].'
)


def build_tagging_prompt(context: str | None = None) -> str:
    """Assemble the full tagging instruction for one image.

    `context` is optional product context (e.g. "Category: Mobile,
    Model: XPERIA10MK5") inferred from the folder path.
    """
    ctx = f"\nProduct context: {context}." if context else ""
    return (
        f"{INSTRUCTION}{ctx}\n\n"
        f"Controlled vocabulary:\n{taxonomy_prompt()}\n\n"
        f"{OUTPUT_FORMAT}"
    )
