"""The shape a search tool returns.

Defined in the core package so both sides can name it without either importing
the other: `tools` produces these, `evidence` consumes them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str = ""
