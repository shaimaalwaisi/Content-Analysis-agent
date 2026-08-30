"""Self-consistency: does the model give the same answer twice?

The test images carry no labels, so nothing here can say whether a tag is
*right*. What it can say is whether the model is reading the image or guessing,
and those are not the same question but they are close enough to be useful: a
model that genuinely sees a side-on photograph of a phone says `side angle`
both times it is asked, while a model with no real evidence for a tag will
often drop or replace it on the second pass.

So each image is tagged twice with a *different* draw of few-shot examples, and
the two tag sets are compared. The score is Jaccard -- the share of tags the
two passes agree on -- which is 1.0 for an identical answer and 0.0 for two
answers with nothing in common.

Two things to be honest about, both worth saying out loud rather than burying:

* Agreement is not accuracy. A model can be confidently, repeatably wrong, and
  this will give it 1.0. It bounds reliability from above, no more.
* It costs a second model call per image, which doubles cost per task. That is
  why it is opt-in (`--consistency`) rather than always on.

The pairing with the labelled set matters: micro-F1 says how right the agent is
on the 8 images that have ground truth, and this says how *stable* it is on the
107 that do not. Neither replaces the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Below this, the two passes disagreed on more than a third of the tags, which
# in practice means at least one tag appeared in only one of them. Those are
# the images worth a human's time; the threshold is a review-budget dial, not
# a statistical claim.
UNSTABLE_BELOW = 0.7


def agreement(first: list[str], second: list[str]) -> float:
    """Jaccard overlap of two tag sets: |shared| / |either|.

    Two empty answers score 1.0 -- they agree, even if they agree on nothing.
    That is the honest reading: the instability worth flagging is the model
    changing its mind, and an image the model twice declined to tag is already
    caught by task success rate.
    """
    a, b = set(first), set(second)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class ConsistencyReport:
    """Per-image agreement, and what it adds up to over a batch."""

    per_image: list[dict] = field(default_factory=list)
    threshold: float = UNSTABLE_BELOW

    @property
    def n(self) -> int:
        return len(self.per_image)

    @property
    def mean_agreement(self) -> float:
        if not self.per_image:
            return 0.0
        return sum(r["agreement"] for r in self.per_image) / self.n

    @property
    def unstable(self) -> list[dict]:
        """The images the two passes disagreed on, worst first."""
        return sorted((r for r in self.per_image
                       if r["agreement"] < self.threshold),
                      key=lambda r: r["agreement"])

    def summary(self) -> str:
        if not self.per_image:
            return "Self-consistency: nothing to compare."
        lines = [
            "Self-consistency (no labels required)",
            f"  Mean agreement : {self.mean_agreement:.3f} over {self.n} "
            f"image(s), tagged twice with different examples",
            f"  Unstable       : {len(self.unstable)} image(s) below "
            f"{self.threshold:.2f}",
        ]
        for row in self.unstable[:5]:
            lines.append(f"      {row['agreement']:.2f}  {row['image']}")
            lines.append(f"            pass 1: {', '.join(row['first']) or '-'}")
            lines.append(f"            pass 2: {', '.join(row['second']) or '-'}")
        if len(self.unstable) > 5:
            lines.append(f"      ... and {len(self.unstable) - 5} more")
        lines.append("  (Agreement is stability, not accuracy: a model can be "
                     "repeatably wrong.)")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "images": self.n,
            "mean_agreement": round(self.mean_agreement, 4),
            "threshold": self.threshold,
            "unstable": len(self.unstable),
            "per_image": [{**r, "agreement": round(r["agreement"], 4)}
                          for r in self.per_image],
        }


def compare_passes(first: list, second: list,
                   threshold: float = UNSTABLE_BELOW) -> ConsistencyReport:
    """Score two runs of the same folder against each other.

    Both arguments are lists of `TagResult`. They are matched on image path
    rather than on position, so a run that skipped or reordered an image
    scores the images the two passes actually share.
    """
    by_path = {r.path: r for r in second}
    rows = []
    for one in first:
        other = by_path.get(one.path)
        if other is None:
            continue
        rows.append({"image": one.path.rsplit("/", 1)[-1],
                     "first": list(one.tags), "second": list(other.tags),
                     "agreement": agreement(one.tags, other.tags)})
    return ConsistencyReport(rows, threshold)
