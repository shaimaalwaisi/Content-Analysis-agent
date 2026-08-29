"""All LLM prompts live here, so wording can be tuned in one place.

There is currently a single prompt: the image-tagging instruction. It is kept
as named text blocks plus a small builder that injects the controlled
vocabulary and the per-image product context.
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
    '["physical design", "side angle", "top"]. No other text.'
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
