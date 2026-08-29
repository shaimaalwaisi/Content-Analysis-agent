"""Tools the agent can call.

A tool is a capability the agent reaches for when the image alone cannot answer
the question: today that means looking a product up to justify `awards`,
`benchmark` or `energy rating`. Backends sit behind one protocol, exactly as
the model clients do in `content_analysis_agent.vlm`:

* MockSearchTool      - offline, deterministic; no key, no network
* AnthropicWebSearch  - Claude's server-side web_search, run on Anthropic's
                        infrastructure, so no second key and no tool loop here

Dependencies point inward: this package imports the core for logging and the
shared `SearchResult` type; the core never imports this one at runtime. The
agent takes a tool as an argument, and the CLI decides which one to pass, so
`content_analysis_agent` stays importable with this folder absent.

The rules turning evidence into tags live in `content_analysis_agent.evidence`,
because that is a statement about the taxonomy rather than about searching.
"""
from .search import (AnthropicWebSearch, MockSearchTool, SearchTool,
                     get_search_tool)
from content_analysis_agent.tools_types import SearchResult

__all__ = ["AnthropicWebSearch", "MockSearchTool", "SearchResult",
           "SearchTool", "get_search_tool"]
