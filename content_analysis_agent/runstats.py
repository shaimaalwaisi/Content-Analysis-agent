"""Workflow metrics for the agent itself, as opposed to tagging quality.

The metrics in `evaluate.py` score predictions against ground-truth labels, so
they only run on the 8 labelled training images. These measure how the agent
*behaves* and need no labels at all, which means they work on the 107
unlabelled test images -- and in production, where labels never exist.

Three groups:

* Hallucination -- how often the model proposes a tag outside the controlled
  vocabulary. `validate_tags` already drops those silently; this counts them.
  It needs no ground truth, which makes it the one quality signal available on
  live traffic.
* Latency -- per-model-call and per-tool-call wall time, as p50/p95 rather than
  a mean, since tail latency is what a user notices.
* Efficiency -- cache hit rate, retries, tool calls and failures per image:
  what a run actually cost to produce.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


@dataclass
class RunStats:
    """Thread-safe counters; run_folder may tag images in parallel."""

    images: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    tool_calls: int = 0
    raw_tags: int = 0
    dropped_tags: int = 0
    images_with_dropped: int = 0
    model_ms: list[float] = field(default_factory=list)
    tool_ms: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- recording -------------------------------------------------------
    def record_image(self) -> None:
        with self._lock:
            self.images += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_model_call(self, ms: float, n_raw: int) -> None:
        with self._lock:
            self.model_calls += 1
            self.model_ms.append(ms)
            self.raw_tags += n_raw

    def record_dropped(self, n: int) -> None:
        with self._lock:
            if n:
                self.dropped_tags += n
                self.images_with_dropped += 1

    def record_tool_call(self, ms: float) -> None:
        with self._lock:
            self.tool_calls += 1
            self.tool_ms.append(ms)

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1

    # ---- derived ---------------------------------------------------------
    @property
    def hallucination_rate(self) -> float:
        """Share of proposed tags that fell outside the vocabulary."""
        return self.dropped_tags / self.raw_tags if self.raw_tags else 0.0

    @property
    def images_hallucinating(self) -> float:
        """Share of model-answered images that proposed at least one bad tag."""
        return (self.images_with_dropped / self.model_calls
                if self.model_calls else 0.0)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.images if self.images else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.images if self.images else 0.0

    def summary(self) -> str:
        return "\n".join([
            "Agent workflow metrics (no labels required)",
            f"  Hallucination   : {self.hallucination_rate:.3f} "
            f"({self.dropped_tags}/{self.raw_tags} proposed tags out of "
            f"vocabulary)",
            f"                    {self.images_hallucinating:.3f} of answered "
            f"images proposed at least one",
            f"  Latency (model) : p50 {_percentile(self.model_ms, 50):.0f} ms | "
            f"p95 {_percentile(self.model_ms, 95):.0f} ms | "
            f"{self.model_calls} call(s)",
            f"  Latency (tool)  : p50 {_percentile(self.tool_ms, 50):.0f} ms | "
            f"p95 {_percentile(self.tool_ms, 95):.0f} ms | "
            f"{self.tool_calls} call(s)",
            f"  Efficiency      : cache hit {self.cache_hit_rate:.3f} | "
            f"{self.retries} retr(ies) | {self.failures} failure(s) "
            f"({self.failure_rate:.3f})",
        ])

    def as_dict(self) -> dict:
        return {
            "images": self.images, "model_calls": self.model_calls,
            "cache_hits": self.cache_hits, "cache_hit_rate":
                round(self.cache_hit_rate, 4),
            "retries": self.retries, "failures": self.failures,
            "failure_rate": round(self.failure_rate, 4),
            "tool_calls": self.tool_calls,
            "raw_tags": self.raw_tags, "dropped_tags": self.dropped_tags,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "images_hallucinating": round(self.images_hallucinating, 4),
            "model_ms_p50": round(_percentile(self.model_ms, 50), 1),
            "model_ms_p95": round(_percentile(self.model_ms, 95), 1),
            "tool_ms_p50": round(_percentile(self.tool_ms, 50), 1),
            "tool_ms_p95": round(_percentile(self.tool_ms, 95), 1),
        }
