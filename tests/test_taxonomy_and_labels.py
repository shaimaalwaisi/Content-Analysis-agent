"""The controlled vocabulary, and the labels encoded in training filenames."""
from agent.labels import (load_labelled,
                                           parse_tags_from_filename)
from agent.taxonomy import (OBSERVED_EXTRA, TAXONOMY,
                                             allowed_tags, normalise)


class TestNormalise:
    def test_lowercases_collapses_whitespace_and_is_idempotent(self):
        assert normalise("Side  Angle") == "side angle"
        assert normalise("  FRONT ANGLE ") == "front angle"
        once = normalise("Multiple   Angles")
        assert normalise(once) == once


class TestAllowedTags:
    def test_contains_generals_and_specifics_but_nothing_invented(self):
        vocab = allowed_tags()
        assert "physical design" in vocab      # a General
        assert "front angle" in vocab          # one of its Specifics
        assert "camera" in vocab               # under feature graphics
        assert "sparkly unicorn" not in vocab
        assert all(normalise(t) == t for t in vocab), "already normalised"

    def test_includes_observed_extras(self):
        # 'left'/'right' appear in training filenames but not in the brief's
        # appendix, so the vocabulary must admit them or the agent would be
        # dropping tags the data itself uses.
        assert OBSERVED_EXTRA and OBSERVED_EXTRA <= allowed_tags()


class TestParseTagsFromFilename:
    def test_reads_the_bracketed_list(self):
        assert parse_tags_from_filename(
            "['physical design', 'side angle', 'top'].jpg"
        ) == ["physical design", "side angle", "top"]

    def test_ignores_directories_and_normalises_what_it_finds(self):
        assert parse_tags_from_filename(
            "/data/train/Mobile/['Physical  Design'].jpg") == [
            "physical design"]

    def test_returns_empty_rather_than_raising_on_anything_else(self):
        for name in ("amazon_co_uk_B09XBQBXS2_Main_Carousel_1.jpg",
                     "['unclosed.jpg", "[not, python].jpg"):
            assert parse_tags_from_filename(name) == []


class TestLoadLabelled:
    def test_finds_every_labelled_image_and_all_labels_are_known(
            self, labelled_dir):
        pairs = load_labelled(labelled_dir)
        assert len(pairs) == 3
        for _path, tags in pairs:
            assert tags and set(tags) <= allowed_tags()


def test_taxonomy_json_is_two_level():
    """Every Specific must sit under a General, so prompts render correctly."""
    assert TAXONOMY
    for general, specifics in TAXONOMY.items():
        assert isinstance(specifics, list) and specifics
        assert normalise(general) == general
