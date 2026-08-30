"""Evaluation against ground-truth tags encoded in the training filenames.

Train images are named like:

    ['physical design', 'side angle', 'top'].jpg

so every training image is a free labelled example. We parse those tags, run
the agent on the same images, and score it on two multi-label metrics:
micro-F1 and macro-F1. They are computed by hand (set arithmetic) to avoid a
heavy sklearn dependency and to make exactly what we measure explicit.

Scoring is leave-one-out: with `--few-shot`, the image being scored is dropped
from its own example list, so the model is never shown the answer it is being
asked for.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.graph import _infer_context, build_graph
from agent.memory import TagMemory
from .runstats import RunStats
from agent.labels import load_labelled
from agent.vlm import VLMClient

if TYPE_CHECKING:
    from tools import SearchTool


# --------------------------- success criteria -------------------------------

# What "good" means, chosen up front so a score can be judged rather than just
# reported. These are targets for the MVP, not measurements.
#
#   1. Beat the label-prior baselines below. A model that cannot outscore
#      "always guess the most common tags" is adding nothing, and on this
#      dataset that floor is high: 'physical design' appears in every label.
#   2. Micro-F1 >= 0.75. Tags feed a human review queue, so pooled per-tag
#      correctness matters more than getting whole sets exactly right.
#   3. Macro-F1 >= 0.40. Micro-F1 is dominated by the handful of tags that
#      appear on almost every image, so it can look healthy while the rare
#      tags -- the ones worth surfacing to a content creator -- are never
#      predicted at all. Macro-F1 weights every tag equally and is the number
#      that falls when that happens.
TARGET_MICRO_F1 = 0.75
TARGET_MACRO_F1 = 0.40


# --------------------------- baselines --------------------------------------

def most_common_tags(truth: list[list[str]], k: int) -> list[str]:
    """The k tags that appear most often across the ground-truth labels."""
    counts = Counter(t for tags in truth for t in tags)
    return [t for t, _ in counts.most_common(k)]


def median_label_size(truth: list[list[str]]) -> int:
    sizes = sorted(len(t) for t in truth)
    return sizes[len(sizes) // 2] if sizes else 0


def baseline_predictions(truth: list[list[str]]) -> dict[str, list[list[str]]]:
    """Model-free predictions to compare the agent against.

    Both are derived from the ground-truth labels themselves, so they are an
    optimistic floor -- they already know the tag distribution the agent has to
    infer from pixels. Beating them is the minimum bar, not a success.
    """
    n = len(truth)
    if not n:
        return {}
    top1 = most_common_tags(truth, 1)
    topk = most_common_tags(truth, median_label_size(truth))
    return {
        f"constant ({', '.join(top1)})": [list(top1)] * n,
        f"prior top-{len(topk)} ({', '.join(topk)})": [list(topk)] * n,
    }


# --------------------------- metrics ---------------------------------------

@dataclass
class Metrics:
    """Two headline numbers, and the per-tag table macro-F1 is averaged from.

    Two, not eight: precision and recall on their own invite reading whichever
    half flatters the run, and Jaccard and exact-match say much the same thing
    as micro-F1 about whole-image overlap. Micro-F1 answers "are the tags
    right", macro-F1 answers "on the rare tags too", and per_tag is where you
    look when either one drops.
    """

    n: int
    micro_f1: float         # pooled over every (image, tag) decision
    macro_f1: float         # mean per-tag F1, every tag weighted equally
    per_tag: dict           # tag -> {support, precision, recall, f1}

    def summary(self) -> str:
        return (
            f"Images evaluated : {self.n}\n"
            f"Micro-F1         : {self.micro_f1:.3f}\n"
            f"Macro-F1         : {self.macro_f1:.3f}"
        )


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_metrics(truth: list[list[str]],
                    pred: list[list[str]]) -> Metrics:
    truth_sets = [set(t) for t in truth]
    pred_sets = [set(p) for p in pred]

    # Micro: pool TP/FP/FN across all images.
    tp = sum(len(t & p) for t, p in zip(truth_sets, pred_sets))
    fp = sum(len(p - t) for t, p in zip(truth_sets, pred_sets))
    fn = sum(len(t - p) for t, p in zip(truth_sets, pred_sets))
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0

    # Per-tag stats, which the macro average is then taken over.
    labels = sorted({t for s in truth_sets for t in s} |
                    {p for s in pred_sets for p in s})
    per_tag: dict = {}
    tag_f1: list[float] = []      # unrounded, so the macro average is exact
    for lab in labels:
        l_tp = sum((lab in t) and (lab in p)
                   for t, p in zip(truth_sets, pred_sets))
        l_fp = sum((lab not in t) and (lab in p)
                   for t, p in zip(truth_sets, pred_sets))
        l_fn = sum((lab in t) and (lab not in p)
                   for t, p in zip(truth_sets, pred_sets))
        support = sum(lab in t for t in truth_sets)
        p = l_tp / (l_tp + l_fp) if (l_tp + l_fp) else 0.0
        r = l_tp / (l_tp + l_fn) if (l_tp + l_fn) else 0.0
        tag_f1.append(_f1(p, r))
        per_tag[lab] = {"support": support, "precision": round(p, 3),
                        "recall": round(r, 3), "f1": round(tag_f1[-1], 3)}

    # Mean of the per-tag F1s, over every tag in the truth or the prediction.
    # Both halves of that matter. Averaging the F1s rather than F1-ing the
    # averaged precision and recall is the standard definition, and the two do
    # not agree. Including tags with no support is what makes the number
    # honest: a tag the model predicts on every image and is never right about
    # scores F1 0.0 and must be averaged in, otherwise a model that spams one
    # wrong in-vocabulary tag still macro-scores a perfect 1.000.
    macro_f1 = sum(tag_f1) / len(tag_f1) if tag_f1 else 0.0

    return Metrics(
        n=len(truth_sets),
        micro_f1=_f1(micro_p, micro_r),
        macro_f1=macro_f1,
        per_tag=per_tag,
    )


def compare_baselines(truth: list[list[str]],
                      pred: list[list[str]]) -> dict[str, Metrics]:
    """Score the agent alongside each model-free baseline."""
    out = {"agent": compute_metrics(truth, pred)}
    for name, guess in baseline_predictions(truth).items():
        out[name] = compute_metrics(truth, guess)
    return out


def failure_warning(records: list[dict]) -> str:
    """Loud warning when images failed, so a broken run cannot read as a bad score.

    Without this an outage, a bad model id, or a provider limit produces a
    confident-looking 0.000 next to the baselines, which is worse than no
    number at all.
    """
    failed = [r for r in records if r.get("error")]
    if not failed:
        return ""
    kinds = {}
    for r in failed:
        kinds[r["error"].split(":")[0]] = kinds.get(
            r["error"].split(":")[0], 0) + 1
    lines = ["", "!" * 62,
             f"WARNING: {len(failed)}/{len(records)} images FAILED and scored "
             f"as empty predictions.",
             "The metrics below understate the model - fix the errors before "
             "reading them.", ""]
    for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:>3} x {kind}")
    example = failed[0]["error"]
    lines.append(f"  first: {example[:150]}")
    lines.append("!" * 62)
    return "\n".join(lines)


def format_comparison(scored: dict[str, Metrics]) -> str:
    """Render the agent-vs-baseline table, with an explicit verdict."""
    width = max(len(n) for n in scored)
    lines = [f"{'':<{width}}   micro-F1   macro-F1",
             "-" * (width + 22)]
    for name, m in scored.items():
        lines.append(f"{name:<{width}}   {m.micro_f1:>8.3f}   "
                     f"{m.macro_f1:>8.3f}")
    agent = scored["agent"]
    best = max((m.micro_f1 for n, m in scored.items() if n != "agent"),
               default=0.0)
    lines.append("")
    lines.append(f"Beats best baseline : "
                 f"{'YES' if agent.micro_f1 > best else 'NO'} "
                 f"({agent.micro_f1:.3f} vs {best:.3f})")
    lines.append(f"Micro-F1 >= {TARGET_MICRO_F1:.2f}    : "
                 f"{'YES' if agent.micro_f1 >= TARGET_MICRO_F1 else 'NO'}")
    lines.append(f"Macro-F1 >= {TARGET_MACRO_F1:.2f}    : "
                 f"{'YES' if agent.macro_f1 >= TARGET_MACRO_F1 else 'NO'}")
    return "\n".join(lines)


def evaluate(root: str, client: VLMClient, sample: int | None = None,
             on_item=None, few_shot: int | None = None,
             memory: TagMemory | None = None,
             search_tool: "SearchTool | None" = None,
             stats: RunStats | None = None) -> tuple[Metrics, list[dict]]:
    """Run the agent over labelled images and score it. Returns metrics plus a
    per-image record (path, truth, predicted) for inspection.

    `few_shot` prepends up to N labelled examples to each request. The image
    being scored is always excluded from its own examples, so the score stays
    honest even though examples and test items share one folder.
    """
    # Local import: fewshot imports this module, so importing it at module
    # level would be circular.
    from agent.fewshot import load_examples

    data = load_labelled(root)
    if sample:
        data = data[:sample]
    app = build_graph(client, memory=memory, search_tool=search_tool,
                      stats=stats)

    truth, pred, records = [], [], []
    for i, (path, gt) in enumerate(data, 1):
        ctx = _infer_context(path)
        if stats:
            stats.record_task()
        examples = (load_examples(root, limit=few_shot, exclude=path)
                    if few_shot else None)
        try:
            out = app.invoke({"image_path": path, "context": ctx or None,
                              "examples": examples})
            got = out.get("tags", [])
            error = None
        except Exception as exc:
            got, error = [], f"{type(exc).__name__}: {exc}"
            if stats:
                stats.record_failure()
            print(f"  ! {os.path.basename(path)}: {exc}")
        if stats:
            stats.record_outcome(len(got))
        truth.append(gt)
        pred.append(got)
        record = {"path": path, "truth": gt, "predicted": got}
        if error:
            record["error"] = error
        records.append(record)
        if on_item:
            on_item(i, len(data), path, gt, got)
    return compute_metrics(truth, pred), records
