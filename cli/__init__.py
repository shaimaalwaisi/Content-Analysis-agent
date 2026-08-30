"""The command-line entry point.

Presentation, like `app/`: it owns no logic of its own, it wires the layers
together. Each subcommand lives in its own module and registers its arguments,
and this package is the only one that imports `tools`, `evaluation` and
`data` alongside the core.

    python -m cli taxonomy
    python -m cli tag --input data/test --provider anthropic --few-shot 8
"""
from .main import main

__all__ = ["main"]
