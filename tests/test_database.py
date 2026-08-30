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

    def test_one_row_per_image_per_run(self, store):
        run = new_run_id()
        for i in range(10):
            store.put(_row(run, name=f"{i}.jpg", tags=["physical design"]))
        assert len(store.rows(run)) == 10
        # Re-tagging an image updates its row rather than adding another.
        store.put(_row(run, name="0.jpg", tags=["physical design", "colour"]))
        rows = store.rows(run)
        assert len(rows) == 10
        assert [r for r in rows if r["image_name"] == "0.jpg"][0]["tags"] == [
            "physical design", "colour"]

    def test_runs_are_listed_newest_first_and_read_by_default(self, store):
        assert store.rows() == [] and store.latest_run() is None
        store.put(_row("20260101-000000", tags=["colour"]))
        store.put(_row("20260102-000000", tags=["front angle"]))
        assert [r["run_id"] for r in store.runs()][0] == "20260102-000000"
        assert store.latest_run() == "20260102-000000"
        assert store.rows()[0]["tags"] == ["front angle"]


class TestGraphWritesRows:
    def test_the_agent_writes_one_row_per_image(self, jpeg, stub, store):
        # StubVLM implements predict_tags only, so the detail columns it
        # cannot fill stay empty rather than being invented.
        run = new_run_id()
        tag_one(build_graph(stub, store=store, run_id=run), jpeg,
                context="Category: Mobile, Model: XPERIA10MK5")
        row, = store.rows(run)
        assert row["tags"] == ["physical design", "front angle"]
        assert row["product"] == "XPERIA10MK5" and row["category"] == "Mobile"
        assert row["model"] == "stub-1" and row["attempts"] == 1
        assert row["description"] == "" and row["specs"] == ""

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

    def test_every_kept_tag_carries_its_reason_and_no_rejected_one_does(
            self, jpeg, store):
        from tests.test_graph import ReasoningVLM
        client = ReasoningVLM(["physical design", "front angle", "sparkly"],
                              ["physical design", "front angle"])
        tag_one(build_graph(client, store=store, run_id="r1"), jpeg)
        row, = store.rows("r1")
        assert row["rationale"] == {"physical design": "because physical "
                                                       "design",
                                    "front angle": "because front angle"}

    def test_a_re_prompted_image_records_both_attempts(self, jpeg, store):
        from tests.test_graph import ReasoningVLM
        client = ReasoningVLM(["sparkly", "unicorn mode"],
                              ["physical design"])
        tag_one(build_graph(client, store=store, run_id="r1"), jpeg)
        row, = store.rows("r1")
        assert row["attempts"] == 2


class TestDetailColumns:
    """Product, category, description and specs -- the columns that used to
    be empty for an upload, which has no folder path to infer them from."""

    def _client(self, **kw):
        from agent.vlm import Prediction

        class DetailVLM:
            model = "detail-1"

            def predict(self, image_b64, media_type, context=None,
                        examples=None, feedback=None):
                return Prediction(["physical design", "front angle"],
                                  {}, **kw)

            def predict_tags(self, *a, **k):
                return ["physical design", "front angle"]

        return DetailVLM()

    def test_the_model_fills_them_when_nothing_else_can(self, jpeg, store):
        client = self._client(category="TV", product="XR-65A95K",
                              description="A 65-inch OLED shown face on.",
                              specs="65-inch, OLED, 4K")
        tag_one(build_graph(client, store=store, run_id="r1"), jpeg)
        row, = store.rows("r1")
        assert row["category"] == "TV" and row["product"] == "XR-65A95K"
        assert row["description"] == "A 65-inch OLED shown face on."
        assert row["specs"] == "65-inch, OLED, 4K"

    def test_the_given_context_beats_the_model(self, jpeg, store):
        # The folder path, or the person at the keyboard, knows the product;
        # the model is reading a name off a photograph.
        client = self._client(category="TV", product="XR-65A95K")
        tag_one(build_graph(client, store=store, run_id="r1"), jpeg,
                context="Category: Mobile, Model: XPERIA1MK5")
        row, = store.rows("r1")
        assert row["category"] == "Mobile" and row["product"] == "XPERIA1MK5"


class TestCachedRowsKeepTheirColumns:
    def test_a_memory_hit_replays_description_and_specs(self, jpeg, store,
                                                        tmp_path):
        from agent.vlm import Prediction

        class DetailVLM:
            model = "detail-1"
            calls = 0

            def predict(self, image_b64, media_type, context=None,
                        examples=None, feedback=None):
                type(self).calls += 1
                return Prediction(["physical design"], {},
                                  category="TV", product="XR-65A95K",
                                  description="A 65-inch OLED, face on.",
                                  specs="65-inch, OLED")

            def predict_tags(self, *a, **k):
                return ["physical design"]

        mem = TagMemory(str(tmp_path / "m.sqlite3"))
        client = DetailVLM()
        app = build_graph(client, memory=mem, store=store, run_id="r1")
        tag_one(app, jpeg)
        app2 = build_graph(client, memory=mem, store=store, run_id="r2")
        tag_one(app2, jpeg)                       # served from memory
        fresh, cached = store.rows("r1")[0], store.rows("r2")[0]
        assert client.calls == 1, "the second run must not call the model"
        assert cached["cached"] is True
        for column in ("description", "specs", "category", "product"):
            assert cached[column] == fresh[column], column
        mem.close()


class TestMigration:
    def test_a_database_without_the_new_columns_is_upgraded(self, tmp_path):
        # Rows written before description and specs existed must stay readable.
        import sqlite3
        path = str(tmp_path / "old.sqlite3")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE taggings (id INTEGER PRIMARY KEY "
                     "AUTOINCREMENT, run_id TEXT NOT NULL, created REAL "
                     "NOT NULL, image_name TEXT NOT NULL, image_path TEXT "
                     "NOT NULL, category TEXT, product TEXT, highlights TEXT "
                     "NOT NULL, tags TEXT NOT NULL, rationale TEXT NOT NULL, "
                     "model TEXT, attempts INTEGER, cached INTEGER)")
        conn.execute("INSERT INTO taggings (run_id, created, image_name, "
                     "image_path, highlights, tags, rationale) VALUES "
                     "('old', 1.0, 'a.jpg', '/a.jpg', '[]', '[\"colour\"]', "
                     "'{}')")
        conn.commit()
        conn.close()

        upgraded = ResultStore(path)
        try:
            row, = upgraded.rows("old")
            assert row["tags"] == ["colour"] and row["description"] is None
            upgraded.put(_row("new", tags=["colour"], description="works"))
            assert upgraded.rows("new")[0]["description"] == "works"
        finally:
            upgraded.close()


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
        # One sqlite connection, four threads writing through it.
        run_folder(labelled_dir, stub, store=store, run_id="r1", workers=4)
        assert len(store.rows("r1")) == 3
