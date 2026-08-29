"""Scoring: the metrics, the baselines, and the guard against a broken run."""
import pytest

from content_analysis_agent.fewshot import load_examples
from evaluation import (baseline_predictions, compare_baselines,
                        compute_metrics, failure_warning, format_comparison,
                        most_common_tags)
from evaluation.runstats import RunStats, _percentile


class TestComputeMetrics:
    def test_perfect_prediction(self):
        truth = [["physical design", "top"], ["colour"]]
        m = compute_metrics(truth, [list(t) for t in truth])
        assert m.micro_f1 == 1.0 and m.exact_match == 1.0 and m.jaccard == 1.0

    def test_nothing_predicted(self):
        m = compute_metrics([["physical design"], ["colour"]], [[], []])
        assert m.micro_precision == 0.0 and m.micro_recall == 0.0
        assert m.micro_f1 == 0.0

    def test_hand_computed_partial_credit(self):
        # 1 TP, 1 FP, 1 FN -> P = R = 0.5, F1 = 0.5, Jaccard = 1/3
        m = compute_metrics([["physical design", "top"]],
                            [["physical design", "colour"]])
        assert m.micro_precision == pytest.approx(0.5)
        assert m.micro_recall == pytest.approx(0.5)
        assert m.micro_f1 == pytest.approx(0.5)
        assert m.jaccard == pytest.approx(1 / 3)
        assert m.exact_match == 0.0

    def test_order_does_not_matter(self):
        a = compute_metrics([["top", "colour"]], [["colour", "top"]])
        assert a.exact_match == 1.0

    def test_per_tag_support_counts_ground_truth(self):
        m = compute_metrics([["colour"], ["colour"], ["top"]],
                            [["colour"], [], ["top"]])
        assert m.per_tag["colour"]["support"] == 2
        assert m.per_tag["colour"]["recall"] == pytest.approx(0.5)


class TestBaselines:
    TRUTH = [["physical design", "front angle"],
             ["physical design", "side angle"],
             ["physical design", "front angle"]]

    def test_most_common_tags_ranks_by_frequency(self):
        assert most_common_tags(self.TRUTH, 1) == ["physical design"]

    def test_constant_baseline_predicts_the_same_thing_every_time(self):
        guesses = baseline_predictions(self.TRUTH)
        constant = next(v for k, v in guesses.items() if k.startswith("constant"))
        assert all(g == constant[0] for g in constant)
        assert len(constant) == len(self.TRUTH)

    def test_the_agent_is_scored_alongside_the_baselines(self):
        scored = compare_baselines(self.TRUTH, [list(t) for t in self.TRUTH])
        assert "agent" in scored and len(scored) == 3
        assert scored["agent"].micro_f1 == 1.0

    def test_a_constant_guess_can_be_hard_to_beat(self):
        # 'physical design' is in every label here, as in the real training
        # set -- which is exactly why a baseline is worth printing.
        scored = compare_baselines(self.TRUTH, [["physical design"]] * 3)
        constant = next(m for k, m in scored.items() if k.startswith("constant"))
        assert scored["agent"].micro_f1 == pytest.approx(constant.micro_f1)

    def test_comparison_table_states_a_verdict(self):
        text = format_comparison(compare_baselines(
            self.TRUTH, [list(t) for t in self.TRUTH]))
        assert "Beats best baseline" in text and "YES" in text


class TestFailureWarning:
    def test_silent_when_every_image_succeeded(self):
        assert failure_warning([{"path": "a.jpg", "truth": [], "predicted": []}]) == ""

    def test_shouts_when_images_failed(self):
        # A broken run must never read as a merely bad score.
        records = [{"path": "a.jpg", "error": "BadRequestError: too many images"},
                   {"path": "b.jpg", "error": "BadRequestError: too many images"},
                   {"path": "c.jpg"}]
        text = failure_warning(records)
        assert "2/3" in text and "FAILED" in text
        assert "BadRequestError" in text


class TestFewShotLeakage:
    def test_an_image_never_sees_its_own_answer(self, labelled_dir):
        from content_analysis_agent.labels import load_labelled
        for path, _tags in load_labelled(labelled_dir):
            examples = load_examples(labelled_dir, exclude=path)
            assert len(examples) == 2, "the scored image must be excluded"

    def test_limit_caps_the_number_of_examples(self, labelled_dir):
        assert len(load_examples(labelled_dir, limit=2)) == 2

    def test_examples_carry_their_tags(self, labelled_dir):
        for _b64, media, tags in load_examples(labelled_dir):
            assert media.startswith("image/") and tags


class TestRunStats:
    def test_percentiles(self):
        values = [100, 200, 300, 400, 900]
        assert _percentile(values, 50) == 300
        assert _percentile(values, 95) == 900
        assert _percentile([], 50) == 0.0

    def test_rates_are_zero_when_nothing_happened(self):
        stats = RunStats()
        assert stats.hallucination_rate == 0.0
        assert stats.cache_hit_rate == 0.0
        assert stats.failure_rate == 0.0

    def test_counters_are_thread_safe(self):
        import threading
        stats = RunStats()

        def record():
            for _ in range(100):
                stats.record_image()

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert stats.images == 1000

    def test_as_dict_is_json_friendly(self):
        import json
        stats = RunStats()
        stats.record_image()
        stats.record_model_call(120.0, 3)
        assert json.loads(json.dumps(stats.as_dict()))["model_calls"] == 1
