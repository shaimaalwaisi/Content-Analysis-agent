"""Marketing tag taxonomy.

Two-level hierarchy (General -> Specific) from Appendix 1 of the brief.
A few Specific tags that appear in the TRAINING filenames but are missing
from Appendix 1 (e.g. 'left', 'right') are added and flagged OBSERVED, so the
model may predict them and evaluation does not unfairly penalise them.

Both General tags (e.g. 'physical design') and Specific tags (e.g. 'side
angle') are valid labels; an image usually gets its General category plus one
or more Specifics.
"""
from __future__ import annotations

TAXONOMY: dict[str, list[str]] = {
    "accessories": ["accessories"],
    "awards": ["awards"],
    "benchmark": ["benchmark"],
    "bonus": ["bonus"],
    "dimension": ["dimension"],
    "energy rating": ["energy rating"],
    "feature graphics": [
        "ai", "application", "battery life", "camera", "compatibility",
        "call quality", "connectivity", "durability", "gaming",
        "picture quality", "portability", "sound quality", "comfort",
        "sustainability", "controls", "customisability",
    ],
    "person": ["social group"],
    "product summary": ["product summary"],
    "physical design": [
        "back angle", "colour", "front angle", "multiple angles", "side angle",
        "earbuds", "case", "top", "bottom",
        "left", "right",  # OBSERVED in train filenames, absent from Appendix 1
    ],
    "usage scene": ["indoor", "outdoor", "transport"],
    "whats in the box": ["whats in the box"],
}

OBSERVED_EXTRA = {"left", "right"}


def allowed_tags() -> set[str]:
    """Flat set of every valid tag (General + Specific)."""
    tags: set[str] = set(TAXONOMY.keys())
    for specifics in TAXONOMY.values():
        tags.update(specifics)
    return tags


def normalise(tag: str) -> str:
    """Lower-case and collapse whitespace: 'Side  Angle' -> 'side angle'."""
    return " ".join(str(tag).lower().split())


def taxonomy_prompt() -> str:
    """Render the hierarchy as text for the model prompt."""
    lines = []
    for general, specifics in TAXONOMY.items():
        if specifics == [general]:
            lines.append(f"- {general}")
        else:
            lines.append(f"- {general}: {', '.join(specifics)}")
    return "\n".join(lines)


if __name__ == "__main__":
    tags = sorted(allowed_tags())
    print(f"{len(tags)} allowed tags:\n")
    print("\n".join(tags))
