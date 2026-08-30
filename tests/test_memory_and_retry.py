"""Memory keying and thread-safety, and what retry does and does not retry."""
import threading

import pytest

from agent.memory import TagMemory, make_key
from agent.retry import call_with_retry, is_transient


class Boom(Exception):
    """Transient by class name, per retry's duck-typed classification."""


class RateLimitError(Exception):
    pass


class BadRequestError(Exception):
    status_code = 400


class ServerError(Exception):
    status_code = 503


@pytest.fixture
def memory(tmp_path):
    mem = TagMemory(str(tmp_path / "m.sqlite3"))
    yield mem
    mem.close()


class TestMakeKey:
    BASE = ("b64data", "model-1", "Category: TV")

    def test_same_inputs_give_the_same_key(self):
        assert make_key(*self.BASE) == make_key(*self.BASE)

    def test_every_input_that_changes_the_answer_changes_the_key(self):
        # Each of these would otherwise replay an answer produced under
        # different conditions -- another model's, another prompt's.
        base = make_key(*self.BASE)
        assert make_key("other", "model-1", "Category: TV") != base
        assert make_key("b64data", "model-2", "Category: TV") != base
        assert make_key("b64data", "model-1", "Category: Mobile") != base
        assert make_key(*self.BASE, extra="enrich") != base

    def test_few_shot_examples_are_part_of_the_key(self):
        a = [("x", "image/jpeg", ["colour"])]
        b = [("y", "image/jpeg", ["colour"])]
        assert make_key(*self.BASE, examples=a) != make_key(*self.BASE)
        assert make_key(*self.BASE, examples=a) != make_key(*self.BASE,
                                                            examples=b)


class TestTagMemory:
    def test_miss_then_hit_and_both_are_counted(self, memory):
        key = make_key("img", "m", None)
        assert memory.get(key) is None                      # miss
        memory.put(key, ["physical design"], "m")
        assert memory.get(key) == ["physical design"]       # hit
        assert (memory.hits, memory.misses) == (1, 1)

    def test_remembers_the_whole_answer_not_just_the_tags(self, memory):
        # A cache hit fills the same results row as a fresh answer, so the
        # description and specs must survive alongside the tags.
        memory.put("k", ["colour"], "m",
                   rationale={"colour": "three finishes"},
                   details={"description": "A phone in three finishes.",
                            "specs": "5000mAh", "category": "Mobile",
                            "product": "XPERIA1MK5"})
        record = memory.get_record("k")
        assert record["tags"] == ["colour"]
        assert record["rationale"] == {"colour": "three finishes"}
        assert record["details"]["specs"] == "5000mAh"
        assert memory.get("k") == ["colour"], "the plain read is unchanged"

    def test_reads_a_cache_written_before_details_existed(self, memory):
        memory.put("k", ["colour"], "m")          # the old shape: a bare list
        record = memory.get_record("k")
        assert record["tags"] == ["colour"]
        assert record["rationale"] == {} and record["details"] == {}

    def test_survives_reopening_the_file(self, tmp_path):
        path = str(tmp_path / "persist.sqlite3")
        first = TagMemory(path)
        first.put(make_key("img", "m", None), ["top"], "m")
        first.close()
        second = TagMemory(path)
        assert second.get(make_key("img", "m", None)) == ["top"]
        second.close()

    def test_concurrent_writes_are_not_lost(self, memory):
        # run_folder tags in parallel, so every write shares one connection.
        def write(i):
            memory.put(make_key(f"img{i}", "m", None), [f"tag{i}"], "m")

        threads = [threading.Thread(target=write, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert memory.size() == 50


class TestIsTransient:
    def test_only_rate_limits_and_server_errors_are_transient(self):
        assert is_transient(ServerError()) and is_transient(RateLimitError())
        assert not is_transient(BadRequestError())
        assert not is_transient(ValueError("nope"))


class TestCallWithRetry:
    def test_returns_on_first_success(self):
        assert call_with_retry(lambda: "ok", sleep=lambda _s: None) == "ok"

    def test_retries_transient_then_succeeds_and_reports_each_retry(self):
        state, seen = {"n": 0}, []

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise RateLimitError("slow down")
            return "ok"

        assert call_with_retry(flaky, on_retry=seen.append,
                               sleep=lambda _s: None) == "ok"
        assert state["n"] == 3
        assert len(seen) == 2, "two failures means two retry callbacks"

    def test_non_transient_raises_immediately(self):
        state = {"n": 0}

        def bad():
            state["n"] += 1
            raise BadRequestError("malformed")

        with pytest.raises(BadRequestError):
            call_with_retry(bad, sleep=lambda _s: None)
        assert state["n"] == 1, "a bad request must not be retried"

    def test_gives_up_after_attempts(self):
        state = {"n": 0}

        def always():
            state["n"] += 1
            raise RateLimitError("still limited")

        with pytest.raises(RateLimitError):
            call_with_retry(always, attempts=3, sleep=lambda _s: None)
        assert state["n"] == 3
