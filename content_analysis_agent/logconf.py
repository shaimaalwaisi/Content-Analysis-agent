"""Structured logging for the agent.

Human progress output stays on stdout (the CLI prints it); this module handles
the machine-readable side. Every event is one JSON object per line, which is
what a log shipper, `jq`, or a monitoring backend expects -- the deployment
story asked for by the brief starts here.

Fields are attached with the standard ``extra=`` argument, so a call reads:

    log.info("tagged", extra={"image": path, "ms": 412, "n_tags": 3})
"""
from __future__ import annotations

import json
import logging
import sys
import time

# Attributes LogRecord always carries; anything else was passed via extra=.
_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None)))
_RESERVED |= {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any extra= fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "WARNING", path: str | None = None) -> None:
    """Configure the package logger.

    `level` is a standard level name; `path` writes JSON lines to a file
    instead of stderr. Called once from the CLI and the Streamlit app.
    """
    logger = logging.getLogger("content_analysis_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
    logger.handlers.clear()
    handler = (logging.FileHandler(path, encoding="utf-8") if path
               else logging.StreamHandler(sys.stderr))
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Module-level logger under the package namespace."""
    return logging.getLogger(f"content_analysis_agent.{name.rsplit('.', 1)[-1]}")
