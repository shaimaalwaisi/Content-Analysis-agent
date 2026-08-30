"""Marketing tag taxonomy (loaded from taxonomy.json).

The controlled vocabulary lives in taxonomy.json next to this module so it can
be edited without touching code. It is a two-level hierarchy (General ->
Specific) from Appendix 1 of the brief. A few Specifics that appear in the
TRAINING filenames but are missing from Appendix 1 (e.g. 'left', 'right') are
listed under 'observed_extra' and merged in, so the model may predict them and
evaluation does not unfairly penalise them.

Public API: TAXONOMY, OBSERVED_EXTRA, allowed_tags(), normalise(),
taxonomy_prompt(), highlight_tags().
"""
from __future__ import annotations

import json
from pathlib import Path

_TAXONOMY_FILE = Path(__file__).with_name("taxonomy.json")

with _TAXONOMY_FILE.open(encoding="utf-8") as _f:
    _data = json.load(_f)

TAXONOMY: dict[str, list[str]] = _data["taxonomy"]
OBSERVED_EXTRA: set[str] = set(_data.get("observed_extra", []))


def allowed_tags() -> set[str]:
    """Flat set of every valid tag (General + Specific + observed extras)."""
    tags: set[str] = set(TAXONOMY.keys())
    for specifics in TAXONOMY.values():
        tags.update(specifics)
    tags.update(OBSERVED_EXTRA)
    return tags


# The General category whose Specifics say what an image is *selling* --
# camera, battery life, durability -- as opposed to how it is framed. The
# results table surfaces these separately, under "Highlights".
HIGHLIGHT_GENERAL = "feature graphics"


def highlight_tags(tags: list[str]) -> list[str]:
    """The selling-point tags among `tags`, in the order given."""
    family = set(TAXONOMY.get(HIGHLIGHT_GENERAL, []))
    return [t for t in tags if normalise(t) in family]


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
