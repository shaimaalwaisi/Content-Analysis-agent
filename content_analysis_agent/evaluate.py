"""Evaluation against ground-truth tags encoded in the training filenames.

Train images are named like:

    ['physical design', 'side angle', 'top'].jpg

so every training image is a free labelled example. We parse those tags, run
the agent on the same images, and report standard multi-label metrics.
Metrics are computed by hand (set arithmetic) to avoid a heavy sklearn
dependency and to make exactly what we measure explicit.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass

from .graph import _infer_context, build_graph
from .memory import TagMemory
from .taxonomy import normalise
from .pipeline import find_images
from .vlm import VLMClient


def parse_tags_from_filename(name: str) -> list[str]:
    """Extract the bracketed tag list from a filename. Returns [] if none."""
    base = os.path.basename(name)
    match = re.search(r"\[.*\]", base, re.DOTALL)
    if not match:
        return []
    try:
        tags = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(tags, (list, tuple)):
        return []
    return [normalise(t) for t in tags]


def load_labelled(root: str) -> list[tuple[str, list[str]]]:
    """(image_path, ground_truth_tags) for every labelled image under root."""
    out = []
    for path in find_images(root):
        tags = parse_tags_from_filename(path)
        if tags:
            out.append((path, tags))
    return out


# --------------------------- metrics ---------------------------------------

@dataclass
class Metrics:
    n: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    jaccard: float          # sample-averaged |intersection| / |union|
    exact_match: float      # fraction of images tagged exactly right
    per_tag: dict           # tag -> {support, precision, recall, f1}

    def summary(self) -> str:
        return (
            f"Images evaluated : {self.n}\n"
            f"Micro  P/R/F1    : {self.micro_precision:.3f} / "
            f"{self.micro_recall:.3f} / {self.micro_f1:.3f}\n"
            f"Macro  P/R/F1    : {self.macro_precision:.3f} / "
            f"{self.macro_recall:.3f} / {self.macro_f1:.3f}\n"
            f"Jaccard (avg)    : {self.jaccard:.3f}\n"
            f"Exact-match      : {self.exact_match:.3f}"
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

    # Per-tag stats, then macro-average over tags seen in ground truth.
    labels = sorted({t for s in truth_sets for t in s} |
                    {p for s in pred_sets for p in s})
    per_tag: dict = {}
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
        per_tag[lab] = {"support": support, "precision": round(p, 3),
                        "recall": round(r, 3), "f1": round(_f1(p, r), 3)}

    seen = [lab for lab in labels if per_tag[lab]["support"] > 0]
    macro_p = sum(per_tag[l]["precision"] for l in seen) / len(seen) if seen else 0.0
    macro_r = sum(per_tag[l]["recall"] for l in seen) / len(seen) if seen else 0.0

    jaccard = sum(len(t & p) / len(t | p) if (t | p) else 1.0
                  for t, p in zip(truth_sets, pred_sets)) / len(truth_sets)
    exact = sum(t == p for t, p in zip(truth_sets, pred_sets)) / len(truth_sets)

    return Metrics(
        n=len(truth_sets),
        micro_precision=micro_p, micro_recall=micro_r, micro_f1=_f1(micro_p, micro_r),
        macro_precision=macro_p, macro_recall=macro_r, macro_f1=_f1(macro_p, macro_r),
        jaccard=jaccard, exact_match=exact, per_tag=per_tag,
    )


def evaluate(root: str, client: VLMClient, sample: int | None = None,
             on_item=None, few_shot: int | None = None,
             memory: TagMemory | None = None) -> tuple[Metrics, list[dict]]:
    """Run the agent over labelled images and score it. Returns metrics plus a
    per-image record (path, truth, predicted) for inspection.

    `few_shot` prepends up to N labelled examples to each request. The image
    being scored is always excluded from its own examples, so the score stays
    honest even though examples and test items share one folder.
    """
    # Local import: fewshot imports this module, so importing it at module
    # level would be circular.
    from .fewshot import load_examples

    data = load_labelled(root)
    if sample:
        data = data[:sample]
    app = build_graph(client, memory=memory)

    truth, pred, records = [], [], []
    for i, (path, gt) in enumerate(data, 1):
        ctx = _infer_context(path)
        examples = (load_examples(root, limit=few_shot, exclude=path)
                    if few_shot else None)
        try:
            out = app.invoke({"image_path": path, "context": ctx or None,
                              "examples": examples})
            got = out.get("tags", [])
        except Exception as exc:
            got = []
            print(f"  ! {os.path.basename(path)}: {exc}")
        truth.append(gt)
        pred.append(got)
        records.append({"path": path, "truth": gt, "predicted": got})
        if on_item:
            on_item(i, len(data), path, gt, got)
    return compute_metrics(truth, pred), records
