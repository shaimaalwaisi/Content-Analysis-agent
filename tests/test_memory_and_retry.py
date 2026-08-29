"""Memory keying and thread-safety, and what retry does and does not retry."""
import threading

import pytest

from content_analysis_agent.memory import TagMemory, make_key
from content_analysis_agent.retry import call_with_retry, is_transient


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

    def test_key_changes_with_the_image(self):
        assert make_key("other", "model-1", "Category: TV") != make_key(*self.BASE)

    def test_key_changes_with_the_model(self):
        # Otherwise switching provider would replay another model's answers.
        assert make_key("b64data", "model-2", "Category: TV") != make_key(*self.BASE)

    def test_key_changes_with_the_context(self):
        assert make_key("b64data", "model-1", "Category: Mobile") != make_key(*self.BASE)

    def test_key_changes_with_few_shot_examples(self):
        examples = [("ex", "image/jpeg", ["physical design"])]
        assert make_key(*self.BASE, examples=examples) != make_key(*self.BASE)

    def test_key_changes_with_enrichment(self):
        # An enriched answer must never be served from a plain run's cache.
        assert make_key(*self.BASE, extra="enrich") != make_key(*self.BASE)

    def test_example_order_and_content_matter(self):
        a = [("x", "image/jpeg", ["colour"])]
        b = [("y", "image/jpeg", ["colour"])]
        assert make_key(*self.BASE, examples=a) != make_key(*self.BASE, examples=b)


class TestTagMemory:
    def test_miss_then_hit(self, memory):
        key = make_key("img", "m", None)
        assert memory.get(key) is None
        memory.put(key, ["physical design"], "m")
        assert memory.get(key) == ["physical design"]

    def test_counts_hits_and_misses(self, memory):
        key = make_key("img", "m", None)
        memory.get(key)                     # miss
        memory.put(key, ["colour"], "m")
        memory.get(key)                     # hit
        assert (memory.hits, memory.misses) == (1, 1)

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
    def test_rate_limits_and_server_errors_are_transient(self):
        assert is_transient(ServerError())
        assert is_transient(RateLimitError())

    def test_client_errors_are_not(self):
        assert not is_transient(BadRequestError())

    def test_unknown_exceptions_are_not(self):
        assert not is_transient(ValueError("nope"))


class TestCallWithRetry:
    def test_returns_on_first_success(self):
        assert call_with_retry(lambda: "ok", sleep=lambda _s: None) == "ok"

    def test_retries_transient_then_succeeds(self):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise RateLimitError("slow down")
            return "ok"

        assert call_with_retry(flaky, sleep=lambda _s: None) == "ok"
        assert state["n"] == 3

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

    def test_on_retry_fires_once_per_retry(self):
        seen = []
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise RateLimitError("x")
            return "ok"

        call_with_retry(flaky, on_retry=seen.append, sleep=lambda _s: None)
        assert len(seen) == 2, "two failures means two retry callbacks"
