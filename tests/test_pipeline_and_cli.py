"""Batch behaviour, and that every CLI subcommand is wired up."""
import json

from content_analysis_agent.pipeline import (find_images, results_to_dicts,
                                             run_folder)
from evaluation.runstats import RunStats

from cli.main import build_parser


class FailFirstVLM:
    model = "fail-1"

    def __init__(self):
        self.calls = 0

    def predict_tags(self, image_b64, media_type, context=None, examples=None):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("bad image")
        return ["physical design"]


def _folder(tmp_path, n=4):
    from PIL import Image
    root = tmp_path / "test" / "TV" / "MODEL-1"
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 32), (i * 10, 0, 0)).save(root / f"img_{i}.jpg")
    return str(tmp_path / "test")


class TestFindImages:
    def test_walks_nested_folders_and_sorts(self, tmp_path):
        paths = find_images(_folder(tmp_path))
        assert len(paths) == 4 and paths == sorted(paths)

    def test_ignores_non_images(self, tmp_path):
        root = _folder(tmp_path)
        (tmp_path / "test" / "notes.txt").write_text("hello")
        assert all(p.endswith(".jpg") for p in find_images(root))


class TestRunFolder:
    def test_tags_every_image(self, tmp_path, stub):
        results = run_folder(_folder(tmp_path), stub)
        assert len(results) == 4
        assert all(r.tags for r in results)

    def test_infers_category_and_model(self, tmp_path, stub):
        first = run_folder(_folder(tmp_path), stub)[0]
        assert first.category == "TV" and first.model == "MODEL-1"

    def test_limit_caps_the_work(self, tmp_path, stub):
        assert len(run_folder(_folder(tmp_path), stub, limit=2)) == 2
        assert stub.calls == 2

    def test_one_bad_image_does_not_abort_the_run(self, tmp_path):
        client = FailFirstVLM()
        results = run_folder(_folder(tmp_path), client)
        assert len(results) == 4
        assert results[0].tags == [] and all(r.tags for r in results[1:])

    def test_failures_are_counted(self, tmp_path):
        stats = RunStats()
        run_folder(_folder(tmp_path), FailFirstVLM(), stats=stats)
        assert stats.failures == 1 and stats.images == 4

    def test_workers_preserve_order(self, tmp_path, stub):
        # Parallelism must never reshuffle results: futures are read in
        # submission order precisely so output stays deterministic.
        folder = _folder(tmp_path)
        serial = [r.path for r in run_folder(folder, stub)]
        parallel = [r.path for r in run_folder(folder, stub, workers=4)]
        assert serial == parallel and len(serial) == 4

    def test_results_serialise_to_json(self, tmp_path, stub):
        records = results_to_dicts(run_folder(_folder(tmp_path), stub, limit=1))
        assert json.loads(json.dumps(records))[0]["category"] == "TV"


class TestCLIParser:
    def test_every_subcommand_is_registered(self):
        parser = build_parser()
        for cmd in ("taxonomy", "tag", "eval", "insights"):
            args = parser.parse_args([cmd] + _minimal_args(cmd))
            assert callable(args.func)

    def test_tag_defaults(self):
        args = build_parser().parse_args(["tag", "--input", "x"])
        assert args.provider == "anthropic" and args.workers == 1
        assert args.few_shot == 0 and args.enrich is False

    def test_shared_flags_are_identical_across_commands(self):
        parser = build_parser()
        tag = parser.parse_args(["tag", "--input", "x"])
        ev = parser.parse_args(["eval", "--train-dir", "x"])
        assert tag.provider == ev.provider
        assert tag.memory == ev.memory
        assert tag.search_tool == ev.search_tool

    def test_insights_requires_a_source(self):
        import pytest
        with pytest.raises(SystemExit):
            build_parser().parse_args(["insights"])

    def test_insights_sources_are_mutually_exclusive(self):
        import pytest
        with pytest.raises(SystemExit):
            build_parser().parse_args(["insights", "--from-sheet",
                                       "--input", "x"])

    def test_log_level_is_global(self):
        args = build_parser().parse_args(["--log-level", "INFO", "taxonomy"])
        assert args.log_level == "INFO"


def _minimal_args(cmd):
    return {"taxonomy": [], "tag": ["--input", "x"],
            "eval": ["--train-dir", "x"], "insights": ["--from-sheet"]}[cmd]
