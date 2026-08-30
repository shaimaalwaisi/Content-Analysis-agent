"""Evaluation: does the agent do its job, and how does it behave doing it?

Two independent kinds of measurement live here, kept apart because they answer
different questions and have different requirements:

* `quality` -- tagging *quality* against ground-truth labels: micro-F1 and
  macro-F1, the per-tag table behind them, plus the model-free baselines and
  success criteria a score is judged against. Needs labels, so it runs only on
  the labelled training images.

* `runstats` -- how the agent *behaves*: task success rate, cost per task and
  latency per action. Needs no labels, so it runs on the unlabelled test set
  and in production.

Names are re-exported **lazily**. `graph` and `pipeline` import `RunStats` from
here to instrument themselves, while `quality` imports `graph` to run the
agent -- so eagerly importing `quality` in this file would close that loop and
fail at import time. PEP 562 module `__getattr__` defers each import until the
name is actually used, which keeps the package a single front door without the
cycle.

Label parsing (`parse_tags_from_filename`, `load_labelled`) deliberately does
NOT live here -- it is in `agent.labels`, because `fewshot` needs it too and
the core package must not import this layer.

The quality module is named `quality`, not `evaluate`: a submodule sets itself
as an attribute of its package on import, so a submodule named `evaluate` would
permanently shadow the `evaluate()` function re-exported here.
"""
from __future__ import annotations

# name -> submodule that defines it
_EXPORTS = {
    "Metrics": "quality",
    "TARGET_MACRO_F1": "quality",
    "TARGET_MICRO_F1": "quality",
    "baseline_predictions": "quality",
    "compare_baselines": "quality",
    "compute_metrics": "quality",
    "evaluate": "quality",
    "failure_warning": "quality",
    "format_comparison": "quality",
    "median_label_size": "quality",
    "most_common_tags": "quality",
    "ACTIONS": "runstats",
    "PRICES_PER_MTOK": "runstats",
    "RunStats": "runstats",
    "price_for": "runstats",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value          # cache, so this runs once per name
    return value


def __dir__():
    return __all__
