"""Argument groups shared by more than one subcommand.

Provider, memory, enrichment and results-database flags were repeated verbatim
across `tag`, `eval` and `insights`, which is how they drift apart -- a default
changed in one place and not the others. Declared once here instead.
"""
from __future__ import annotations

import argparse

from agent.memory import DEFAULT_PATH as MEMORY_PATH
from agent.vlm import PROVIDERS

# The default lives in the tools layer, but reading a constant does not make
# the CLI depend on the tool: the store itself is imported only when built.
from tools.database import DEFAULT_PATH as RESULTS_PATH


def add_provider_args(p: argparse.ArgumentParser) -> None:
    # One choice, kept as a flag so every run record still says what was
    # called -- a settings block that omits the provider is harder to read a
    # year later than one that states the obvious.
    p.add_argument("--provider", default="anthropic", choices=PROVIDERS,
                   help="the vision model to call (Claude)")
    p.add_argument("--model", default=None,
                   help="override the Claude model id (default: "
                        "claude-haiku-4-5-20251001)")


def add_memory_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--memory", default=MEMORY_PATH,
                   help="path to the agent's tag memory (SQLite)")
    p.add_argument("--no-memory", action="store_true",
                   help="ignore stored tags and always call the model")


def add_enrich_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--enrich", action="store_true",
                   help="look the product up to justify non-visual tags "
                        "(awards, benchmark, energy rating)")
    p.add_argument("--search-tool", default="mock",
                   choices=["mock", "anthropic"],
                   help="search backend for --enrich")


def add_results_db_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=RESULTS_PATH,
                   help="results database the UI reads (SQLite)")
    p.add_argument("--no-db", action="store_true",
                   help="do not record this run in the results database")


def build_memory(args):
    """The TagMemory a command should use, or None when disabled."""
    from agent.memory import TagMemory
    return None if args.no_memory else TagMemory(args.memory)


def build_search_tool(args):
    """The search tool a command should use, or None when not enriching."""
    if not getattr(args, "enrich", False):
        return None
    from tools import get_search_tool
    return get_search_tool(args.search_tool)


def build_store(args):
    """The ResultStore a command should write to, or None when disabled."""
    if getattr(args, "no_db", False):
        return None
    from tools import ResultStore
    return ResultStore(getattr(args, "db", RESULTS_PATH))
