"""Tools the agent can call.

Four of them:

* search    -- looking a product up when the image alone cannot answer the
               question, which is what justifies `awards`, `benchmark` or
               `energy rating`
* database  -- where the tagged rows land, and what the results table reads
* scraper   -- where the images come from when nobody uploads any: one Sony
               page in, a `Category/Model` folder of new pictures out
* imagemeta -- what a file says about itself: size, format, EXIF, and the
               perceptual hash the scraper dedupes on

The vision model is not in here: it is the agent's own reasoning, not a
capability it reaches out to. Nor is classification -- the folder a picture
lands in already names its category, so nothing has to be classified back out
of the pixels.

Backends sit behind one protocol, exactly as the model clients do in
`agent.vlm`:

* MockSearchTool      - offline, deterministic; no key, no network
* AnthropicWebSearch  - Claude's server-side web_search, run on Anthropic's
                        infrastructure, so no second key and no tool loop here
* MockScraper         - offline, deterministic; draws its own images
* SonyScraper         - live: reads a page, resolves image URLs, downloads

Dependencies point inward: this package imports the core for logging and the
shared `SearchResult` type; the core never imports this one at runtime. The
agent takes a tool as an argument, and the CLI decides which one to pass, so
`agent` stays importable with this folder absent.

The rules turning evidence into tags live in `agent.enrichment`,
because that is a statement about the taxonomy rather than about searching.
"""
from .database import DEFAULT_PATH as RESULTS_PATH
from .database import ResultStore, Tagging, new_run_id
from .imagemeta import ImageMeta, hamming, perceptual_hash, read_metadata
from .scraper import (Fetched, MockScraper, PageBlocked, ScraperTool,
                      SonyScraper, category_for, get_scraper, image_urls,
                      product_for)
from .search import (AnthropicWebSearch, MockSearchTool, SearchTool,
                     get_search_tool)
from agent.enrichment import SearchResult

__all__ = ["AnthropicWebSearch", "Fetched", "ImageMeta", "MockScraper",
           "MockSearchTool", "PageBlocked", "RESULTS_PATH", "ResultStore",
           "ScraperTool",
           "SearchResult", "SearchTool", "SonyScraper", "Tagging",
           "category_for", "get_scraper", "get_search_tool", "hamming",
           "image_urls", "new_run_id", "perceptual_hash", "product_for",
           "read_metadata"]
