"""Argument parser assembly and dispatch.

Each subcommand owns its own module and registers itself through `add_parser`,
so adding one means adding a file and a line here rather than editing a single
300-line parser.
"""
from __future__ import annotations

import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from agent.logconf import setup_logging

from . import evaluate, insights, tag, taxonomy

COMMANDS = (taxonomy, tag, evaluate, insights)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli",
        description="Annotate product images with marketing tags.")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="structured (JSON lines) log verbosity")
    p.add_argument("--log-file", default=None,
                   help="write JSON log lines here instead of stderr")
    p.add_argument("--results-dir", default="results",
                   help="where each run's JSON record is written")
    p.add_argument("--no-results", action="store_true",
                   help="do not write a run record")
    sub = p.add_subparsers(dest="cmd", required=True)
    for command in COMMANDS:
        command.add_parser(sub)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    args.func(args)
    return 0
