"""Enrichment: turning what a search found into tags the image cannot show.

Named for the `--enrich` flag and the graph's `_enrich` node, which are the
two things that use it. Some tags in the taxonomy are simply not visual -- no
photograph reveals that a product won an `award`, appeared in a `benchmark`,
or carries an `energy rating` -- so the agent looks the product up, and this
module decides which of those tags the retrieved text actually earns.

Kept in the core package rather than with the search backends, deliberately:
the rules are a statement about *this* taxonomy, and `tools.search` is meant
to stay generic enough to be shared with other agents (an MCP tool, one day),
which have no interest in Sony marketing tags. Searching and interpreting are
two jobs, and only the first is worth sharing. It also means `graph` can turn
evidence into tags without importing the `tools` layer at runtime.

`SearchResult` lives here for the same reason: it is the shape the two sides
agree on -- `tools.search` produces them, `tags_from_evidence` below consumes
them -- and defining it beside its consumer, in a package `tools` already
depends on, keeps the dependency pointing one way without a third module
existing only to hold a dataclass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str = ""

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
