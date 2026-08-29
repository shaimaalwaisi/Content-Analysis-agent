"""Which non-visual tags a piece of evidence supports.

Kept in the core package rather than with the search backends: this is a
statement about the taxonomy, not about searching. It also means `graph` can
turn evidence into tags without importing the `tools` layer at runtime, so the
agent package stays importable on its own.
"""
from __future__ import annotations

import re

from .tools_types import SearchResult

# Non-visual tags the enrich step can contribute, and the evidence that earns
# each one. Keyword matching is deliberate: it keeps the decision inspectable
# and cheap, and it never invents a tag outside this table.
EVIDENCE_RULES: dict[str, tuple[str, ...]] = {
    "awards": ("award", "winner", "best of", "editors choice", "editor's choice",
               "accolade", "prize", "recognised", "recognized"),
    "benchmark": ("benchmark", "test results", "lab test", "measured",
                  "score of", "rtings", "displaymate"),
    "energy rating": ("energy rating", "energy label", "energy class",
                      "energy efficiency", "eu energy"),
}


def tags_from_evidence(results: list[SearchResult]) -> list[str]:
    """Which non-visual tags the evidence supports.

    Matching is on whole words so 'awards' is not fired by 'towards'.
    """
    haystack = " ".join(f"{r.title} {r.snippet}" for r in results).lower()
    found = []
    for tag, keywords in EVIDENCE_RULES.items():
        for kw in keywords:
            if re.search(rf"(?<!\w){re.escape(kw)}", haystack):
                found.append(tag)
                break
    return found
