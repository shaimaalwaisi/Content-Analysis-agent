"""One JSON record per run, written to results/.

Every invocation of `tag`, `eval` or `insights` leaves a timestamped file
behind, so a run is reproducible after the terminal has scrolled away: what was
asked for, what came back, and what it cost. `eval --report` still writes
wherever you point it; this is the automatic copy.

The settings block records only the arguments that change the outcome, so two
files can be diffed to see why two runs differ.
"""
from __future__ import annotations

import glob
import json
import os
import time

DEFAULT_DIR = "results"

# Arguments worth recording: everything that changes what a run produces.
_SETTINGS = ("provider", "model", "few_shot", "workers", "limit", "enrich",
             "search_tool", "no_memory", "sample", "input", "train_dir",
             "results", "metadata", "from_sheet", "synthetic", "metric",
             "min_support")


def settings_of(args) -> dict:
    """The subset of parsed arguments that affects the result."""
    settings = {name: getattr(args, name) for name in _SETTINGS
                if getattr(args, name, None) not in (None, False)}
    if settings.get("from_sheet"):
        # This mode reads tags out of the sheet's filenames and calls no
        # model, so recording argparse's provider default would imply one ran.
        for name in ("provider", "model", "workers", "limit"):
            settings.pop(name, None)
    return settings


def write_run(command: str, args, payload: dict,
              results_dir: str | None = None) -> str | None:
    """Write one run record and return its path, or None when disabled.

    Never raises: a run that produced good tags should not fail at the last
    step because a directory is read-only.
    """
    if getattr(args, "no_results", False):
        return None
    directory = results_dir or getattr(args, "results_dir", None) or DEFAULT_DIR
    record = {
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": settings_of(args),
        **payload,
    }
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory, f"{time.strftime('%Y%m%d-%H%M%S')}_{command}.json")
        with open(path, "w") as handle:
            json.dump(record, handle, indent=2, default=str)
    except OSError as exc:
        print(f"  ! could not write the run record: {exc}")
        return None
    return path


def latest(command: str | None = None,
           results_dir: str = DEFAULT_DIR) -> str | None:
    """Path of the most recent run record, optionally for one command."""
    pattern = f"*_{command}.json" if command else "*.json"
    matches = sorted(glob.glob(os.path.join(results_dir, pattern)))
    return matches[-1] if matches else None
