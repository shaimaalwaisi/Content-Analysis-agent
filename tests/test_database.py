"""The database tool: what a run leaves behind for the content creator."""
import pytest

from agent.graph import build_graph, tag_one
from agent.memory import TagMemory
from agent.pipeline import run_folder
from tools.database import ResultStore, Tagging, new_run_id


@pytest.fixture
def store(tmp_path):
    s = ResultStore(str(tmp_path / "results.sqlite3"))
    yield s
    s.close()


def _row(run_id, name="a.jpg", **kw):
    return Tagging(run_id=run_id, image_path=f"/data/Mobile/XPERIA/{name}",
                   **kw)


class TestStore:
    def test_stores_and_reads_back_a_row(self, store):
        run = new_run_id()
        store.put(_row(run, tags=["physical design", "camera"],
                       highlights=["camera"], rationale={"camera": "a lens"},
                       category="Mobile", product="XPERIA"))
        row, = store.rows(run)
        assert row["image_name"] == "a.jpg"
        assert row["tags"] == ["physical design", "camera"]
        assert row["highlights"] == ["camera"]
        assert row["rationale"] == {"camera": "a lens"}
        assert row["category"] == "Mobile" and row["product"] == "XPERIA"

    def test_ten_images_are_ten_rows(self, store):
        run = new_run_id()
        for i in range(10):
            store.put(_row(run, name=f"{i}.jpg", tags=["physical design"]))
        assert len(store.rows(run)) == 10

    def test_re_tagging_an_image_updates_its_row(self, store):
        run = new_run_id()
        store.put(_row(run, tags=["physical design"]))
        store.put(_row(run, tags=["physical design", "colour"]))
        rows = store.rows(run)
        assert len(rows) == 1, "one image in one run is one row"
        assert rows[0]["tags"] == ["physical design", "colour"]

    def test_runs_are_listed_newest_first_and_read_by_default(self, store):
        store.put(_row("20260101-000000", tags=["colour"]))
        store.put(_row("20260102-000000", tags=["front angle"]))
        assert [r["run_id"] for r in store.runs()][0] == "20260102-000000"
        assert store.latest_run() == "20260102-000000"
        assert store.rows()[0]["tags"] == ["front angle"]

    def test_an_empty_database_reads_as_no_rows(self, store):
        assert store.rows() == [] and store.latest_run() is None


class TestGraphWritesRows:
    def test_the_agent_writes_one_row_per_image(self, jpeg, stub, store):
        run = new_run_id()
        tag_one(build_graph(stub, store=store, run_id=run), jpeg,
                context="Category: Mobile, Model: XPERIA10MK5")
        row, = store.rows(run)
        assert row["tags"] == ["physical design", "front angle"]
        assert row["product"] == "XPERIA10MK5" and row["category"] == "Mobile"
        assert row["model"] == "stub-1" and row["attempts"] == 1

    def test_a_memory_hit_still_produces_a_row(self, jpeg, stub, store,
                                               tmp_path):
        mem = TagMemory(str(tmp_path / "m.sqlite3"))
        app = build_graph(stub, memory=mem, store=store, run_id="r1")
        tag_one(app, jpeg)
        tag_one(app, jpeg)               # served from memory
        row, = store.rows("r1")
        assert stub.calls == 1 and row["cached"] is True, \
            "the creator still expects the image in the table"
        mem.close()

    def test_highlights_are_the_selling_point_tags(self, jpeg, stub, store):
        stub.tags = ["feature graphics", "camera", "front angle"]
        tag_one(build_graph(stub, store=store, run_id="r1"), jpeg)
        row, = store.rows("r1")
        assert row["highlights"] == ["camera"]

    def test_the_reason_for_each_kept_tag_is_stored(self, jpeg, store):
        from tests.test_graph import ReasoningVLM
        client = ReasoningVLM(["physical design", "front angle"])
        tag_one(build_graph(client, store=store, run_id="r1"), jpeg)
        row, = store.rows("r1")
        assert row["rationale"] == {"physical design": "because physical "
                                                       "design",
                                    "front angle": "because front angle"}

    def test_a_rejected_tag_carries_no_reason(self, jpeg, store):
        from tests.test_graph import ReasoningVLM
        client = ReasoningVLM(["physical design", "front angle", "sparkly"])
        tag_one(build_graph(client, store=store, run_id="r1"), jpeg)
        row, = store.rows("r1")
        assert "sparkly" not in row["rationale"]

    def test_a_re_prompted_image_records_both_attempts(self, jpeg, store):
        from tests.test_graph import ReasoningVLM
        client = ReasoningVLM(["sparkly", "unicorn mode"],
                              ["physical design"])
        tag_one(build_graph(client, store=store, run_id="r1"), jpeg)
        row, = store.rows("r1")
        assert row["attempts"] == 2


class TestBatch:
    def test_a_folder_run_fills_the_table(self, labelled_dir, stub, store):
        run = new_run_id()
        results = run_folder(labelled_dir, stub, store=store, run_id=run)
        rows = store.rows(run)
        assert len(rows) == len(results) == 3
        assert {r["image_name"] for r in rows} == {
            r.path.rsplit("/", 1)[-1] for r in results}

    def test_parallel_workers_share_one_connection(self, labelled_dir, stub,
                                                   store):
        run_folder(labelled_dir, stub, store=store, run_id="r1", workers=4)
        assert len(store.rows("r1")) == 3
