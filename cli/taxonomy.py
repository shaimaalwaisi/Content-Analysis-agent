"""`taxonomy` -- print the controlled vocabulary."""
from __future__ import annotations

from agent.taxonomy import allowed_tags, taxonomy_prompt


def run(_args) -> None:
    print(taxonomy_prompt())
    print(f"\n{len(sorted(allowed_tags()))} allowed tags total.")


def add_parser(sub) -> None:
    p = sub.add_parser("taxonomy", help="print the controlled tag vocabulary")
    p.set_defaults(func=run)
