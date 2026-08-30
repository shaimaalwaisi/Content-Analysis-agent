"""The agent itself: validation, memory routing, enrichment, instrumentation."""
import pytest

from agent.graph import _infer_context, build_graph, tag_one
from agent.memory import TagMemory
from agent.enrichment import SearchResult
from evaluation.runstats import RunStats


class FakeSearch:
    def __init__(self, results=None, boom=False):
        self.results = results or []
        self.boom = boom
        self.calls = 0

    def search(self, query):
        self.calls += 1
        if self.boom:
            raise RuntimeError("search is down")
        return self.results


AWARD_EVIDENCE = [SearchResult("r", "Winner of the EISA award for best TV.")]


class TestInferContext:
    def test_reads_category_and_model_from_the_path(self):
        ctx = _infer_context("data/test/TV/XR-65A95K/img.jpg")
        assert "Category: TV" in ctx and "Model: XR-65A95K" in ctx
        assert "Category: Video & Sound" in _infer_context(
            "data/test/Video & Sound/WH-CH520/img.jpg")
        # An upload lands in a temp file, so there is nothing to infer.
        assert _infer_context("/tmp/tmpabc123.jpg") == ""


class TestValidation:
    def test_normalises_deduplicates_and_drops_the_unknown(self, jpeg, stub):
        stub.tags = ["Physical  Design", "sparkly unicorn", "FRONT ANGLE",
                     "physical design"]
        assert tag_one(build_graph(stub), jpeg) == ["physical design",
                                                    "front angle"]

    def test_a_wholly_invalid_answer_yields_no_tags(self, jpeg, stub):
        stub.tags = ["feature graphics: camera"]     # a real failure mode
        assert tag_one(build_graph(stub), jpeg) == []


class TestMemoryRouting:
    def test_second_identical_request_skips_the_model(self, jpeg, stub, tmp_path):
        mem = TagMemory(str(tmp_path / "m.sqlite3"))
        app = build_graph(stub, memory=mem)
        stub.tags = ["physical design", "sparkly unicorn"]
        first = tag_one(app, jpeg)
        second = tag_one(app, jpeg)
        # Identical answers, one model call -- and what was stored is the
        # validated list, not the raw one the model returned.
        assert first == second == ["physical design"]
        assert stub.calls == 1, "a cache hit must not call the model"
        mem.close()

    def test_different_context_is_a_different_entry(self, jpeg, stub, tmp_path):
        mem = TagMemory(str(tmp_path / "m.sqlite3"))
        app = build_graph(stub, memory=mem)
        tag_one(app, jpeg, context="Category: TV")
        tag_one(app, jpeg, context="Category: Mobile")
        assert stub.calls == 2
        mem.close()

    def test_without_memory_every_call_reaches_the_model(self, jpeg, stub):
        app = build_graph(stub)
        tag_one(app, jpeg)
        tag_one(app, jpeg)
        assert stub.calls == 2


class TestEnrichment:
    def test_adds_non_visual_tags_the_image_cannot_show(self, jpeg, stub):
        app = build_graph(stub, search_tool=FakeSearch(AWARD_EVIDENCE))
        assert "awards" in tag_one(app, jpeg, context="Category: TV")

    def test_cannot_inject_a_tag_outside_the_vocabulary(self, jpeg, stub):
        # Whatever a tool returns, validate_tags still decides.
        rogue = [SearchResult("r", "sparkly unicorn award winner")]
        tags = tag_one(build_graph(stub, search_tool=FakeSearch(rogue)),
                       jpeg, context="Category: TV")
        assert "sparkly unicorn" not in tags

    def test_a_search_outage_keeps_the_model_tags(self, jpeg, stub):
        app = build_graph(stub, search_tool=FakeSearch(boom=True))
        assert tag_one(app, jpeg, context="Category: TV") == [
            "physical design", "front angle"]

    def test_no_context_means_no_search(self, jpeg, stub):
        tool = FakeSearch(AWARD_EVIDENCE)
        tag_one(build_graph(stub, search_tool=tool), jpeg, context="")
        assert tool.calls == 0, "there is nothing to look up without a product"

    def test_enriched_and_plain_results_cache_separately(self, jpeg, stub, tmp_path):
        mem = TagMemory(str(tmp_path / "m.sqlite3"))
        plain = build_graph(stub, memory=mem)
        enriched = build_graph(stub, memory=mem,
                               search_tool=FakeSearch(AWARD_EVIDENCE))
        a = tag_one(plain, jpeg, context="Category: TV")
        b = tag_one(enriched, jpeg, context="Category: TV")
        assert "awards" not in a and "awards" in b
        assert tag_one(plain, jpeg, context="Category: TV") == a
        mem.close()


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


class TestReasoningLoop:
    def test_a_weak_answer_is_asked_again_and_a_good_one_is_not(self, jpeg):
        weak = ReasoningVLM(["sparkly", "unicorn mode", "colour"],
                            ["physical design", "colour"])
        assert tag_one(build_graph(weak), jpeg) == ["physical design",
                                                    "colour"]
        assert weak.calls == 2
        good = ReasoningVLM(["physical design", "front angle"])
        tag_one(build_graph(good), jpeg)
        assert good.calls == 1, "a clean answer must not cost a second call"

    def test_the_loop_is_bounded(self, jpeg):
        client = ReasoningVLM(["sparkly", "unicorn mode"])
        assert tag_one(build_graph(client), jpeg) == []
        assert client.calls == 2, "the loop must stop, however bad the answer"

    def test_the_second_prompt_names_the_rejected_tags(self, jpeg):
        # An empty answer is weak too, so this also covers the re-prompt after
        # the model returns nothing at all.
        client = ReasoningVLM(["sparkly", "unicorn mode", "colour"],
                              ["physical design"])
        tag_one(build_graph(client), jpeg)
        first, second = client.feedback_seen
        assert first is None
        assert "sparkly" in second and "unicorn mode" in second
        assert "colour" not in second, "only rejected tags are fed back"


class TestInstrumentation:
    def test_a_weak_answer_costs_a_second_model_call(self, jpeg, stub):
        stub.tags = ["physical design", "sparkly", "unicorn mode"]
        stats = RunStats()
        stats.record_task()
        # Two bad tags out of three is a weak answer, so the reasoning loop
        # asks a second time -- and this stub answers the same way twice. The
        # re-prompt is why latency is measured per action and not per task.
        tag_one(build_graph(stub, stats=stats), jpeg)
        assert stats.model_calls == 2
        assert stats.latency_per_action()["model"]["calls"] == 2

    def test_passes_one_turns_the_loop_off_and_encode_is_timed(self, jpeg,
                                                               stub):
        # StubVLM implements predict_tags only -- no reasons, no feedback --
        # so this also covers a plain client running the graph end to end.
        stub.tags = ["physical design", "sparkly", "unicorn mode"]
        stats = RunStats()
        stats.record_task()
        assert tag_one(build_graph(stub, stats=stats, passes=1), jpeg) == [
            "physical design"]
        assert stats.model_calls == 1
        assert stats.latency_per_action()["encode"]["calls"] == 1

    def test_a_cache_hit_costs_no_model_call(self, jpeg, stub, tmp_path):
        stub.tags = ["physical design", "sparkly"]
        mem = TagMemory(str(tmp_path / "m.sqlite3"))
        stats = RunStats()
        app = build_graph(stub, memory=mem, stats=stats)
        stats.record_task()
        tag_one(app, jpeg)
        stats.record_task()
        tag_one(app, jpeg)
        # Two tasks, one model answer: the second was free, which is the whole
        # argument for the cache and shows up directly in cost per task.
        assert stats.cache_hits == 1 and stats.model_calls == 1
        mem.close()

    def test_examples_reach_the_client(self, jpeg, stub):
        examples = [("b64", "image/jpeg", ["physical design"])]
        tag_one(build_graph(stub), jpeg, examples=examples)
        assert stub.seen_examples[0] == examples
