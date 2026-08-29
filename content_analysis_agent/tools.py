"""Search tools for the enrich step.

Some tags in the taxonomy are not visual. No amount of looking at a product
photo reveals whether the model won an `award`, appeared in a `benchmark`, or
carries an `energy rating` -- that knowledge lives outside the image. The
enrich node closes that gap by looking the product up.

Two backends ship, behind one protocol, mirroring `vlm.py`:

* MockSearchTool      - offline, deterministic; no key, no network
* AnthropicWebSearch  - Claude's server-side web_search tool: the search runs
                        on Anthropic's infrastructure, so there is no separate
                        search provider, key, or client-side execution loop

Whatever a tool returns is only ever a *suggestion*: enrichment adds candidate
tags, and `validate_tags` still decides what survives. A tool cannot put a tag
outside the controlled vocabulary into the output.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .logconf import get_logger

log = get_logger(__name__)

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


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str = ""


class SearchTool(Protocol):
    def search(self, query: str) -> list[SearchResult]:
        ...


@dataclass
class MockSearchTool:
    """Deterministic offline stand-in.

    Returns fixed snippets for a couple of well-known Sony models so the enrich
    path can be demonstrated and tested without a key. It is a fixture, not a
    search engine: never use it to claim anything about a real product.
    """

    corpus: dict[str, list[SearchResult]] = field(default_factory=lambda: {
        "xr-65a95k": [
            SearchResult("Sony A95K review",
                         "Winner of the EISA Best OLED TV award; benchmark "
                         "measurements put peak brightness above 1000 nits.",
                         "https://example.invalid/a95k"),
            SearchResult("A95K energy label",
                         "Carries an EU energy rating of G in HDR mode.",
                         "https://example.invalid/a95k-energy"),
        ],
        "wh-ch720n": [
            SearchResult("WH-CH720N review",
                         "Lab test results show effective noise cancellation "
                         "for the price.", "https://example.invalid/ch720n"),
        ],
    })

    def search(self, query: str) -> list[SearchResult]:
        key = query.lower()
        for model, results in self.corpus.items():
            if model in key:
                return results
        return []


@dataclass
class AnthropicWebSearch:
    """Claude's server-side web search.

    The search executes on Anthropic's infrastructure and the results come back
    in the same response, so there is no second API key and no tool-execution
    loop to write here.
    """

    model: str = "claude-sonnet-5"
    max_uses: int = 3
    max_tokens: int = 1024
    allowed_domains: list[str] | None = None

    def __post_init__(self) -> None:
        from anthropic import Anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or add it to a .env "
                "file in the repo root.")
        self._client = Anthropic()

    def search(self, query: str) -> list[SearchResult]:
        tool: dict = {"type": "web_search_20260209", "name": "web_search",
                      "max_uses": self.max_uses}
        if self.allowed_domains:
            tool["allowed_domains"] = self.allowed_domains
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, tools=[tool],
            messages=[{"role": "user", "content": query}])

        out: list[SearchResult] = []
        for block in resp.content:
            if getattr(block, "type", "") != "web_search_tool_result":
                continue
            content = block.content
            # A successful result is a list; an error is a single object.
            if not isinstance(content, list):
                log.warning("web_search_error", extra={
                    "error": getattr(content, "error_code", str(content))})
                continue
            for item in content:
                out.append(SearchResult(
                    title=getattr(item, "title", "") or "",
                    snippet=(getattr(item, "encrypted_content", "") or
                             getattr(item, "page_age", "") or "")[:500],
                    url=getattr(item, "url", "") or ""))
        # The model's own prose summarises what it found; keep it as evidence
        # since search results are often returned encrypted.
        text = " ".join(b.text for b in resp.content
                        if getattr(b, "type", "") == "text")
        if text:
            out.append(SearchResult("summary", text[:1000]))
        return out


def get_search_tool(name: str, model: str | None = None) -> SearchTool:
    name = (name or "mock").lower()
    if name == "mock":
        return MockSearchTool()
    if name in ("anthropic", "web"):
        return AnthropicWebSearch(model=model or "claude-sonnet-5")
    raise ValueError(f"Unknown search tool: {name!r}. Choose: mock, anthropic")


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
