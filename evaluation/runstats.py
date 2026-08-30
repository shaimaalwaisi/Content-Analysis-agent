"""Workflow metrics for the agent itself, as opposed to tagging quality.

These measure how the agent *behaves*, and need no ground-truth labels, which
means they work on the unlabelled test images -- and in production, where
labels never exist.

Three metrics, one per question an operator actually asks:

* Task success rate -- of the images we were asked to tag, how many came back
  with usable tags? A task is one image. It fails if the agent raised, and it
  fails if it returned nothing: an empty tag list is a task that did not do its
  job, however cleanly it finished.
* Cost per task -- input and output tokens priced at the provider's published
  rate and divided by tasks. This is the number that decides whether tagging a
  100k-image catalogue is affordable, and it is why the memory cache exists: a
  cache hit is a completed task that cost nothing.
* Latency per action -- wall time per *action* the agent takes (model call,
  search call, image encode), as p50/p95 rather than a mean, because tail
  latency is what a user notices. Per action rather than per task, because a
  task that re-prompts once takes two model calls and the per-task number hides
  which action is slow.

Nothing here measures whether a tag is *correct* -- that needs ground truth,
and this module deliberately has none. Out-of-vocabulary predictions are still
logged by the graph (`out_of_vocab_tags`), where they belong: they are an
observability signal, not one of the three.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

# USD per million tokens, (input, output), by model id prefix -- longest match
# wins, so a dated id like 'claude-haiku-4-5-20251001' prices off its family.
# Anthropic list prices as published June 2026. A model that is not in here is
# reported as unpriced rather than guessed at: a made-up cost per task is worse
# than an admitted blank, because it looks like a measurement.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Actions worth timing separately, in the order a run performs them.
ACTIONS = ("encode", "model", "search")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def price_for(model_id: str) -> tuple[float, float] | None:
    """(input, output) USD per million tokens, or None if we cannot price it."""
    matches = [k for k in PRICES_PER_MTOK if model_id.startswith(k)]
    if not matches:
        return None
    return PRICES_PER_MTOK[max(matches, key=len)]


@dataclass
class RunStats:
    """Thread-safe counters; run_folder may tag images in parallel."""

    model_id: str = ""
    tasks: int = 0
    successes: int = 0
    failures: int = 0
    cache_hits: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    action_ms: dict[str, list[float]] = field(
        default_factory=lambda: {a: [] for a in ACTIONS})
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- recording -------------------------------------------------------
    def record_task(self) -> None:
        """One image handed to the agent."""
        with self._lock:
            self.tasks += 1

    def record_action(self, action: str, ms: float) -> None:
        with self._lock:
            self.action_ms.setdefault(action, []).append(ms)

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_model_call(self, ms: float, input_tokens: int = 0,
                          output_tokens: int = 0) -> None:
        with self._lock:
            self.model_calls += 1
            self.action_ms.setdefault("model", []).append(ms)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def record_tool_call(self, ms: float) -> None:
        with self._lock:
            self.action_ms.setdefault("search", []).append(ms)

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_outcome(self, n_tags: int) -> None:
        """The end of one task. Tags means success; nothing means failure."""
        with self._lock:
            if n_tags:
                self.successes += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1

    # ---- the three metrics ----------------------------------------------
    @property
    def task_success_rate(self) -> float:
        """Share of tasks that returned at least one valid tag."""
        return self.successes / self.tasks if self.tasks else 0.0

    @property
    def cost_usd(self) -> float | None:
        """What the run's model calls cost, or None if the model is unpriced."""
        rate = price_for(self.model_id)
        if rate is None:
            return None
        return (self.input_tokens * rate[0] +
                self.output_tokens * rate[1]) / 1_000_000

    @property
    def cost_per_task(self) -> float | None:
        total = self.cost_usd
        if total is None or not self.tasks:
            return total if total is None else 0.0
        return total / self.tasks

    def latency_per_action(self) -> dict[str, dict]:
        """action -> {calls, p50, p95} in milliseconds, actions actually used."""
        return {name: {"calls": len(times),
                       "p50": round(_percentile(times, 50), 1),
                       "p95": round(_percentile(times, 95), 1)}
                for name, times in self.action_ms.items() if times}

    def summary(self) -> str:
        cost = self.cost_per_task
        # No tokens means no spend whatever the model was, so that check comes
        # first: a fully cached run costing nothing must not read as unpriced.
        cached = (f", {self.cache_hits} of {self.tasks} served from cache"
                  if self.cache_hits else "")
        if not (self.input_tokens or self.output_tokens):
            money = "$0 (no tokens billed -- every task cached, or the mock)"
        elif cost is None:
            money = (f"unpriced ({self.model_id or 'no model id recorded'} is "
                     f"not in the price table)")
        else:
            money = (f"${cost:.5f} | ${self.cost_usd:.4f} over "
                     f"{self.tasks} task(s) | {self.input_tokens} in / "
                     f"{self.output_tokens} out tokens{cached}")
        lines = [
            "Agent workflow metrics (no labels required)",
            f"  Task success   : {self.task_success_rate:.3f} "
            f"({self.successes}/{self.tasks} tasks returned tags; "
            f"{self.failures} error(s), {self.retries} retr(ies))",
            f"  Cost per task  : {money}",
            "  Latency/action :",
        ]
        latency = self.latency_per_action()
        if not latency:
            lines.append("      (no actions timed)")
        for name, at in latency.items():
            lines.append(f"      {name:<7} p50 {at['p50']:>7.0f} ms | "
                         f"p95 {at['p95']:>7.0f} ms | {at['calls']} call(s)")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        cost = self.cost_per_task
        return {
            # the three
            "task_success_rate": round(self.task_success_rate, 4),
            "cost_per_task_usd": None if cost is None else round(cost, 6),
            "latency_per_action_ms": self.latency_per_action(),
            # what they are computed from
            "tasks": self.tasks, "successes": self.successes,
            "failures": self.failures, "retries": self.retries,
            "model_id": self.model_id, "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6),
            "cache_hits": self.cache_hits,
        }
