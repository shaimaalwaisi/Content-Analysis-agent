"""Tools the agent can call.

Two of them:

* search   -- looking a product up when the image alone cannot answer the
              question, which is what justifies `awards`, `benchmark` or
              `energy rating`
* database -- where the tagged rows land, and what the results table reads

The vision model is not in here: it is the agent's own reasoning, not a
capability it reaches out to.

Search backends sit behind one protocol, exactly as the model clients do in
`agent.vlm`:

* MockSearchTool      - offline, deterministic; no key, no network
* AnthropicWebSearch  - Claude's server-side web_search, run on Anthropic's
                        infrastructure, so no second key and no tool loop here

Dependencies point inward: this package imports the core for logging and the
shared `SearchResult` type; the core never imports this one at runtime. The
agent takes a tool as an argument, and the CLI decides which one to pass, so
`agent` stays importable with this folder absent.

The rules turning evidence into tags live in `agent.enrichment`,
because that is a statement about the taxonomy rather than about searching.
"""
from .database import DEFAULT_PATH as RESULTS_PATH
from .database import ResultStore, Tagging, new_run_id
from .search import (AnthropicWebSearch, MockSearchTool, SearchTool,
                     get_search_tool)
from agent.enrichment import SearchResult

__all__ = ["AnthropicWebSearch", "MockSearchTool", "RESULTS_PATH",
           "ResultStore", "SearchResult", "SearchTool", "Tagging",
           "get_search_tool", "new_run_id"]
