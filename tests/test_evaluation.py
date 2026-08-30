"""Scoring against the labels, and the three workflow metrics."""
import pytest

from agent.fewshot import load_examples
from evaluation import (baseline_predictions, compare_baselines,
                        compute_metrics, failure_warning, format_comparison,
                        most_common_tags)
from evaluation.consistency import (ConsistencyReport, agreement,
                                    compare_passes)
from evaluation.runstats import RunStats, _percentile


class TestComputeMetrics:
    def test_perfect_and_empty_predictions(self):
        truth = [["physical design", "top"], ["colour"]]
        perfect = compute_metrics(truth, [list(t) for t in truth])
        assert perfect.micro_f1 == 1.0 and perfect.macro_f1 == 1.0
        nothing = compute_metrics(truth, [[], []])
        assert nothing.micro_f1 == 0.0 and nothing.macro_f1 == 0.0

    def test_hand_computed_partial_credit(self):
        # 1 TP, 1 FP, 1 FN -> micro P = R = F1 = 0.5. Three tags are involved
        # and only 'physical design' is ever right, so macro-F1 = 1/3.
        m = compute_metrics([["physical design", "top"]],
                            [["physical design", "colour"]])
        assert m.micro_f1 == pytest.approx(0.5)
        assert m.macro_f1 == pytest.approx(1 / 3)

    def test_order_does_not_matter_and_support_counts_the_truth(self):
        assert compute_metrics([["top", "colour"]],
                               [["colour", "top"]]).micro_f1 == 1.0
        m = compute_metrics([["colour"], ["colour"], ["top"]],
                            [["colour"], [], ["top"]])
        assert m.per_tag["colour"]["support"] == 2
        assert m.per_tag["colour"]["recall"] == pytest.approx(0.5)
        # Macro is the mean of the per-tag F1s, to the 3 decimals per_tag
        # rounds to (it is averaged from the unrounded values).
        assert m.macro_f1 == pytest.approx(
            sum(t["f1"] for t in m.per_tag.values()) / len(m.per_tag),
            abs=1e-3)

    def test_macro_f1_punishes_a_spammed_wrong_tag(self):
        # Why macro averages over tags with no support too: every true tag is
        # predicted perfectly, but 'gaming' is bolted onto every image and is
        # never right. Micro-F1 sees the false positives; macro must too, or
        # it reports a flawless 1.0 for a model that spams.
        truth = [["colour"], ["colour"], ["colour"], ["awards"]]
        pred = [t + ["gaming"] for t in truth]
        m = compute_metrics(truth, pred)
        assert m.micro_f1 == pytest.approx(2 / 3)
        assert m.per_tag["gaming"]["support"] == 0
        assert m.macro_f1 == pytest.approx(2 / 3)


class TestBaselines:
    TRUTH = [["physical design", "front angle"],
             ["physical design", "side angle"],
             ["physical design", "front angle"]]

    def test_a_constant_guess_can_be_hard_to_beat(self):
        # 'physical design' is in every label here, as in the real training
        # set -- which is exactly why a baseline is worth printing.
        assert most_common_tags(self.TRUTH, 1) == ["physical design"]
        guesses = baseline_predictions(self.TRUTH)
        constant_guess = next(v for k, v in guesses.items()
                              if k.startswith("constant"))
        assert len(constant_guess) == len(self.TRUTH)
        assert all(g == constant_guess[0] for g in constant_guess)
        scored = compare_baselines(self.TRUTH, [["physical design"]] * 3)
        constant = next(m for k, m in scored.items()
                        if k.startswith("constant"))
        assert len(scored) == 3
        assert scored["agent"].micro_f1 == pytest.approx(constant.micro_f1)

    def test_the_comparison_table_states_a_verdict(self):
        text = format_comparison(compare_baselines(
            self.TRUTH, [list(t) for t in self.TRUTH]))
        assert "Beats best baseline" in text and "YES" in text


class TestFailureWarning:
    def test_silent_on_success_and_loud_on_failure(self):
        # A run that failed must not read as a run that scored badly.
        assert failure_warning([{"path": "a.jpg"}]) == ""
        text = failure_warning([{"path": "a.jpg", "error": "APIError: 500"},
                                {"path": "b.jpg"}])
        assert "1/2 images FAILED" in text and "APIError" in text


class TestFewShotLeakage:
    def test_an_image_never_sees_its_own_answer(self, labelled_dir):
        from agent.labels import load_labelled
        for path, _tags in load_labelled(labelled_dir):
            examples = load_examples(labelled_dir, exclude=path)
            assert len(examples) == 2, "the scored image must be excluded"

    def test_limit_caps_the_examples_and_each_carries_its_tags(
            self, labelled_dir):
        assert len(load_examples(labelled_dir, limit=2)) == 2
        for _b64, media, tags in load_examples(labelled_dir):
            assert media.startswith("image/") and tags


class TestRunStats:
    def test_percentiles(self):
        values = [100, 200, 300, 400, 900]
        assert _percentile(values, 50) == 300
        assert _percentile(values, 95) == 900
        assert _percentile([], 50) == 0.0

    def test_counters_are_thread_safe(self):
        import threading
        stats = RunStats()

        def record():
            for _ in range(100):
                stats.record_task()

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert stats.tasks == 1000

    def test_a_task_that_returns_nothing_is_not_a_success(self):
        import json
        empty = RunStats()
        assert empty.task_success_rate == 0.0
        assert empty.cost_per_task is None      # no model id, so no price
        assert empty.latency_per_action() == {}

        stats = RunStats()
        for n_tags in (2, 0, 3):
            stats.record_task()
            stats.record_outcome(n_tags)
        stats.record_model_call(120.0)
        assert stats.successes == 2
        assert stats.task_success_rate == pytest.approx(2 / 3)
        assert json.loads(json.dumps(stats.as_dict()))["model_calls"] == 1

    def test_cost_per_task_prices_a_dated_model_id(self):
        # 'claude-haiku-4-5-20251001' must price off the 'claude-haiku-4-5'
        # family: $1/MTok in, $5/MTok out.
        stats = RunStats(model_id="claude-haiku-4-5-20251001")
        stats.record_task()
        stats.record_task()
        stats.record_model_call(100.0, input_tokens=1_000_000,
                                output_tokens=200_000)
        assert stats.cost_usd == pytest.approx(2.0)      # 1.00 + 5 * 0.2
        assert stats.cost_per_task == pytest.approx(1.0)

    def test_an_unpriced_model_reports_no_cost_rather_than_zero(self):
        # A guessed price reads as a measurement; None reads as "we do not
        # know", which is the truth for a provider we have no rate card for.
        stats = RunStats(model_id="some-local-llava")
        stats.record_task()
        stats.record_model_call(100.0, input_tokens=5000, output_tokens=50)
        assert stats.cost_usd is None and stats.cost_per_task is None
        assert stats.as_dict()["cost_per_task_usd"] is None
        assert "unpriced" in stats.summary()

    def test_latency_is_reported_per_action(self):
        stats = RunStats()
        stats.record_action("encode", 5.0)
        stats.record_model_call(300.0)
        stats.record_model_call(900.0)
        stats.record_tool_call(150.0)
        latency = stats.latency_per_action()
        assert set(latency) == {"encode", "model", "search"}
        assert latency["model"]["calls"] == 2
        assert latency["model"]["p95"] == 900.0
        assert latency["encode"]["p50"] == 5.0


class _Result:
    """The two fields compare_passes reads off a TagResult."""

    def __init__(self, path, tags):
        self.path, self.tags = path, tags


class TestSelfConsistency:
    def test_agreement_is_jaccard_over_the_two_tag_sets(self):
        assert agreement(["a", "b"], ["b", "a"]) == 1.0
        assert agreement(["a", "b"], ["a"]) == pytest.approx(0.5)
        assert agreement(["a"], ["b"]) == 0.0
        # Two answers of nothing agree. An image the model twice declined to
        # tag is a task-success failure, not an instability.
        assert agreement([], []) == 1.0

    def test_passes_are_matched_on_path_not_position(self):
        first = [_Result("/x/a.jpg", ["colour"]),
                 _Result("/x/b.jpg", ["top"])]
        second = [_Result("/x/b.jpg", ["top"]),          # reordered
                  _Result("/x/a.jpg", ["colour"])]
        report = compare_passes(first, second)
        assert report.n == 2 and report.mean_agreement == 1.0
        # An image missing from the second pass is skipped, not scored 0.0:
        # it never disagreed, it simply never answered.
        assert compare_passes(first, second[:1]).n == 1

    def test_unstable_images_are_listed_worst_first(self):
        first = [_Result("/x/a.jpg", ["colour", "top"]),
                 _Result("/x/b.jpg", ["colour"]),
                 _Result("/x/c.jpg", ["colour"])]
        second = [_Result("/x/a.jpg", ["colour", "top"]),   # 1.0, stable
                  _Result("/x/b.jpg", ["top"]),             # 0.0
                  _Result("/x/c.jpg", ["colour", "top"])]   # 0.5
        report = compare_passes(first, second)
        assert report.mean_agreement == pytest.approx(0.5)
        assert [r["image"] for r in report.unstable] == ["b.jpg", "c.jpg"]

    def test_the_summary_refuses_to_claim_accuracy(self):
        report = compare_passes([_Result("/x/a.jpg", ["colour"])],
                                [_Result("/x/a.jpg", ["top"])])
        text = report.summary()
        assert "not accuracy" in text, "stability is not correctness"
        assert "a.jpg" in text, "an unstable image is named, to be reviewed"

    def test_as_dict_is_json_friendly_and_an_empty_report_says_so(self):
        import json
        assert ConsistencyReport().mean_agreement == 0.0
        assert "nothing to compare" in ConsistencyReport().summary()
        report = compare_passes([_Result("/x/a.jpg", ["colour"])],
                                [_Result("/x/a.jpg", ["colour", "top"])])
        payload = json.loads(json.dumps(report.as_dict()))
        assert payload["images"] == 1
        assert payload["mean_agreement"] == 0.5
        assert payload["per_image"][0]["second"] == ["colour", "top"]


class TestSeededExamples:
    def test_a_seed_repeats_exactly_and_no_seed_changes_nothing(
            self, labelled_dir):
        # A consistency run has to be reproducible, or the two passes cannot
        # be compared with anything later.
        assert (load_examples(labelled_dir, limit=2, seed=17) ==
                load_examples(labelled_dir, limit=2, seed=17))
        assert load_examples(labelled_dir) == load_examples(labelled_dir)

    def test_some_seed_changes_the_draw(self, labelled_dir):
        # The second pass must be able to ask a different question. Which
        # seed does it is not the point -- that one exists is.
        def tags(**kw):
            return [tuple(t) for *_, t in load_examples(labelled_dir, **kw)]

        plain = tags(limit=2)
        assert any(tags(limit=2, seed=s) != plain for s in range(10))
