"""Argument groups shared by more than one subcommand.

Provider, memory and enrichment flags were repeated verbatim across `tag`,
`eval` and `insights`, which is how they drift apart -- a default changed in
one place and not the others. Declared once here instead.
"""
from __future__ import annotations

import argparse

from content_analysis_agent.memory import DEFAULT_PATH as MEMORY_PATH
from content_analysis_agent.vlm import PROVIDERS


def add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="anthropic", choices=PROVIDERS,
                   help="which vision model to call")
    p.add_argument("--model", default=None, help="override the model id")


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


def build_memory(args):
    """The TagMemory a command should use, or None when disabled."""
    from content_analysis_agent.memory import TagMemory
    return None if args.no_memory else TagMemory(args.memory)


def build_search_tool(args):
    """The search tool a command should use, or None when not enriching."""
    if not getattr(args, "enrich", False):
        return None
    from tools import get_search_tool
    return get_search_tool(args.search_tool)
