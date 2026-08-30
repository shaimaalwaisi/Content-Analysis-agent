"""Fixtures shared by the test suite.

Two rules the whole suite follows:

* No network. Every test drives the agent with a stub client defined here --
  `StubVLM`, or `ReasoningVLM` where the test needs reasons and feedback -- so
  `pytest` never spends money, needs no API key, and never fails because a
  provider is down. Nothing in the suite constructs a real client.
* No dependency on the dataset. The image files are gitignored, so fixtures
  synthesise images rather than reading `data/`; a fresh clone can run the
  tests immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from PIL import Image


def _write_image(path, size=(64, 64), fmt="JPEG", colour=(200, 30, 30)):
    Image.new("RGB", size, colour).save(path, format=fmt)
    return str(path)


@pytest.fixture
def jpeg(tmp_path):
    """A small, genuine JPEG."""
    return _write_image(tmp_path / "small.jpg")


@pytest.fixture
def big_jpeg(tmp_path):
    """Larger than MAX_IMAGE_DIM, so it must be downscaled."""
    return _write_image(tmp_path / "big.jpg", size=(2000, 1500))


@pytest.fixture
def png_named_jpg(tmp_path):
    """A PNG with a .jpg name -- the training set contains one of these."""
    return _write_image(tmp_path / "liar.jpg", fmt="PNG")


@dataclass
class StubVLM:
    """A client that returns fixed tags and counts how often it was called."""

    tags: list = field(default_factory=lambda: ["physical design",
                                                "front angle"])
    model: str = "stub-1"
    calls: int = 0
    seen_examples: list = field(default_factory=list)

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        self.calls += 1
        self.seen_examples.append(examples)
        return list(self.tags)


@pytest.fixture
def stub():
    return StubVLM()


class ReasoningVLM:
    """A client that reasons: it returns a tag list plus a why for each, and
    can be told to answer differently once it has had feedback."""

    model = "reasoning-1"

    def __init__(self, first, second=None):
        self.first, self.second = first, second or first
        self.calls = 0
        self.feedback_seen = []

    def predict(self, image_b64, media_type, context=None, examples=None,
                feedback=None):
        from agent.vlm import Prediction
        self.calls += 1
        self.feedback_seen.append(feedback)
        tags = self.first if feedback is None else self.second
        return Prediction(list(tags), {t: f"because {t}" for t in tags})

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        return self.predict(image_b64, media_type, context, examples).tags


@pytest.fixture
def reasoning():
    """The ReasoningVLM class itself, so a test can construct one with the
    answers it needs. It lives here rather than in a test module because two
    files want it, and a test module is not importable from another one --
    `tests` is deliberately not a package, so cross-importing raised
    ModuleNotFoundError at collection time.
    """
    return ReasoningVLM


@pytest.fixture
def labelled_dir(tmp_path):
    """A miniature training set: filenames encode their own tags."""
    root = tmp_path / "train" / "Mobile" / "XPERIA"
    root.mkdir(parents=True)
    for tags in (["physical design", "front angle"],
                 ["physical design", "side angle"],
                 ["physical design", "colour"]):
        _write_image(root / f"{tags}.jpg")
    return str(tmp_path / "train")
