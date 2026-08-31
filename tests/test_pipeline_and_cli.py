"""Batch behaviour, and that every CLI subcommand is wired up."""
import json

import pytest

from agent.pipeline import (find_images, results_to_dicts,
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
    def test_walks_nested_folders_sorted_and_ignores_non_images(self, tmp_path):
        root = _folder(tmp_path)
        (tmp_path / "test" / "notes.txt").write_text("hello")
        paths = find_images(root)
        assert len(paths) == 4 and paths == sorted(paths)
        assert all(p.endswith(".jpg") for p in paths)


class TestRunFolder:
    def test_tags_every_image_and_infers_category_and_model(self, tmp_path,
                                                            stub):
        results = run_folder(_folder(tmp_path), stub)
        assert len(results) == 4 and all(r.tags for r in results)
        assert results[0].category == "TV" and results[0].model == "MODEL-1"

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
        assert stats.failures == 1 and stats.tasks == 4
        # The failed image returned no tags, so 3 of 4 tasks succeeded.
        assert stats.successes == 3
        assert stats.task_success_rate == pytest.approx(0.75)

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
        for cmd in ("taxonomy", "fetch", "tag", "insights"):
            args = parser.parse_args([cmd] + _minimal_args(cmd))
            assert callable(args.func)

    def test_tag_defaults_and_the_log_level_is_global(self):
        args = build_parser().parse_args(["tag", "--input", "x"])
        assert args.provider == "anthropic" and args.workers == 1
        assert args.few_shot == 0 and args.enrich is False
        assert build_parser().parse_args(
            ["--log-level", "INFO", "taxonomy"]).log_level == "INFO"

    def test_shared_flags_are_identical_across_commands(self):
        # add_provider_args and friends exist so the flags cannot drift; with
        # `tag` the only model-calling command left, `insights --input` is
        # what still shares them.
        parser = build_parser()
        tag = parser.parse_args(["tag", "--input", "x"])
        ins = parser.parse_args(["insights", "--input", "x"])
        assert tag.provider == ins.provider
        assert tag.memory == ins.memory

    def test_insights_needs_exactly_one_source(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["insights"])
        with pytest.raises(SystemExit):
            build_parser().parse_args(["insights", "--from-sheet",
                                       "--input", "x"])


def _minimal_args(cmd):
    return {"taxonomy": [], "tag": ["--input", "x"],
            "fetch": ["--url", "https://sony.com/bravia/xr-65a95k"],
            "insights": ["--from-sheet"]}[cmd]


class TestRunRecords:
    """Every run leaves a JSON record behind in results/."""

    def _args(self, tmp_path, **overrides):
        parser = build_parser()
        args = parser.parse_args(["tag", "--input", "x"])
        args.results_dir = str(tmp_path / "results")
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_a_record_carries_the_payload_command_and_settings(self, tmp_path):
        from cli.runlog import write_run
        path = write_run("tag", self._args(tmp_path), {"images": 3})
        assert path and path.endswith("_tag.json")
        record = json.load(open(path))
        assert record["images"] == 3 and record["command"] == "tag"
        assert record["started_at"]
        assert record["settings"]["provider"] == "anthropic"

    def test_writing_is_optional_and_never_kills_a_run(self, tmp_path):
        # A run that produced good tags must not fail at the last step,
        # whether records are switched off or the directory is unwritable.
        from cli.runlog import write_run
        assert write_run("tag", self._args(tmp_path, no_results=True),
                         {}) is None
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory")
        args = self._args(tmp_path)
        args.results_dir = str(blocker)
        assert write_run("tag", args, {}) is None

    def test_latest_finds_the_newest_record_or_nothing(self, tmp_path):
        from cli.runlog import latest, write_run
        args = self._args(tmp_path)
        write_run("tag", args, {"n": 1})
        newest = write_run("tag", args, {"n": 2})
        assert latest("tag", results_dir=str(tmp_path / "results")) == newest
        assert latest("tag", results_dir=str(tmp_path / "empty")) is None

    def test_from_sheet_records_no_provider(self, tmp_path):
        # Nothing calls a model in that mode; naming one would mislead.
        from cli.runlog import settings_of
        args = build_parser().parse_args(["insights", "--from-sheet"])
        assert "provider" not in settings_of(args)


class TestDatabasePathsAreOverridable:
    """The container puts both SQLite files on a volume, and does it through
    the environment rather than by passing flags to every command."""

    def test_results_and_memory_read_their_env_vars(self, tmp_path,
                                                    monkeypatch):
        import importlib

        import agent.memory
        import tools.database
        monkeypatch.setenv("RESULTS_DB", str(tmp_path / "r.sqlite3"))
        monkeypatch.setenv("MEMORY_DB", str(tmp_path / "m.sqlite3"))
        try:
            assert importlib.reload(tools.database).DEFAULT_PATH \
                == str(tmp_path / "r.sqlite3")
            assert importlib.reload(agent.memory).DEFAULT_PATH \
                == str(tmp_path / "m.sqlite3")
        finally:
            # Other tests import these modules; leave them as they were.
            monkeypatch.delenv("RESULTS_DB")
            monkeypatch.delenv("MEMORY_DB")
            importlib.reload(tools.database)
            importlib.reload(agent.memory)

    def test_the_defaults_are_unchanged_without_them(self):
        import agent.memory
        import tools.database
        assert tools.database.DEFAULT_PATH == "results.sqlite3"
        assert agent.memory.DEFAULT_PATH == ".agent_memory.sqlite3"
