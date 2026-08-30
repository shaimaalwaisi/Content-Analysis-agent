"""The textual half of the prompt, so wording can be tuned in one place.

There is currently a single instruction: the image-tagging prompt, kept as
named text blocks plus a small builder that injects the controlled vocabulary,
the per-image product context, a reason per tag when asked, and -- on a second
attempt -- what went wrong the first time.

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


# The four facts a results table needs that the tag vocabulary cannot express:
# what the product is, and what the image says about it. Asked for in the same
# call as the tags, because a second call would double the cost of a run to
# fill four columns.
DETAIL_FIELDS = ("Category", "Model", "Description", "Specs")

# Reasoning mode: the same answer, preceded by the model's own justification
# for each tag and the details above. The JSON array stays the last thing on
# the page -- and nothing before it may contain a bracket -- so the parser that
# reads plain answers reads these unchanged: the extras are additive, and a
# model that ignores them still produces a valid answer.
REASONED_OUTPUT_FORMAT = (
    "Work in three steps.\n"
    "Step 1 - write one line for EVERY tag you will use, Generals and "
    "Specifics alike: the tag, a dash, and why the image supports it in at "
    "most 12 words. Example:\n"
    "physical design - the product itself is the subject\n"
    "side angle - the phone is photographed from its edge\n"
    "Step 2 - then these four lines, exactly these labels, in this order:\n"
    "Category: one of Mobile, TV, Video & Sound - or unknown\n"
    "Model: the model name printed on the image or its infographic, e.g. "
    "XPERIA1MK5, XR-65A95K - or unknown if none is shown\n"
    "Description: one sentence, at most 20 words, saying what the image "
    "shows\n"
    "Specs: figures and names you can literally READ in the image, "
    "comma-separated, e.g. 5000mAh, ZEISS, 65-inch, IP68 - or none. Never "
    "guess a specification the image does not state.\n"
    "Step 3 - on the final line, and nothing after it, give ONLY the JSON "
    'array of those same tags, e.g. ["physical design", "side angle"].\n'
    "Each array element must be exactly one tag, copied verbatim from the "
    "vocabulary. Never join a General category to its Specific in one string: "
    'write ["feature graphics", "camera"], not ["feature graphics: camera"]. '
    "Use no square brackets anywhere before that final line."
)


def build_feedback(dropped: list[str], kept: list[str]) -> str:
    """Tell the model what its previous answer got wrong.

    Only ever names tags that failed validation, so the feedback cannot push
    the model towards a particular answer -- it can only push it back inside
    the vocabulary.
    """
    if not dropped:
        return ("Your previous answer contained no usable tag. Look again and "
                "choose at least one tag from the vocabulary.")
    listed = ", ".join(f"{t!r}" for t in dropped[:8])
    tail = (f" You kept {len(kept)} valid tag(s)." if kept
            else " None of your tags were valid.")
    return (f"A previous attempt on this image proposed {listed}, which "
            f"is not in the controlled vocabulary and was discarded.{tail} "
            f"Answer again using only vocabulary tags, copied verbatim.")


def build_tagging_prompt(context: str | None = None, reasons: bool = False,
                         feedback: str | None = None) -> str:
    """Assemble the full tagging instruction for one image.

    `context` is optional product context (e.g. "Category: Mobile,
    Model: XPERIA10MK5") inferred from the folder path. `reasons` asks the
    model to justify each tag before answering. `feedback` is what a previous
    attempt got wrong (see build_feedback) and is appended last, where it is
    hardest to ignore.
    """
    ctx = f"\nProduct context: {context}." if context else ""
    fmt = REASONED_OUTPUT_FORMAT if reasons else OUTPUT_FORMAT
    note = f"\n\n{feedback}" if feedback else ""
    return (
        f"{INSTRUCTION}{ctx}\n\n"
        f"Controlled vocabulary:\n{taxonomy_prompt()}\n\n"
        f"{fmt}{note}"
    )
