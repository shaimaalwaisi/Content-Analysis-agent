"""The controlled vocabulary, and the labels encoded in training filenames."""
from content_analysis_agent.labels import (load_labelled,
                                           parse_tags_from_filename)
from content_analysis_agent.taxonomy import (OBSERVED_EXTRA, TAXONOMY,
                                             allowed_tags, normalise)


class TestNormalise:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalise("Side  Angle") == "side angle"
        assert normalise("  FRONT ANGLE ") == "front angle"

    def test_is_idempotent(self):
        once = normalise("Multiple   Angles")
        assert normalise(once) == once


class TestAllowedTags:
    def test_contains_generals_and_specifics(self):
        vocab = allowed_tags()
        assert "physical design" in vocab      # a General
        assert "front angle" in vocab          # one of its Specifics
        assert "camera" in vocab               # under feature graphics

    def test_includes_observed_extras(self):
        # 'left'/'right' appear in training filenames but not in the brief's
        # appendix; evaluation must not penalise the model for predicting them.
        assert OBSERVED_EXTRA
        assert OBSERVED_EXTRA <= allowed_tags()

    def test_every_tag_is_normalised_already(self):
        assert all(normalise(t) == t for t in allowed_tags())

    def test_rejects_invented_tags(self):
        assert "sparkly unicorn" not in allowed_tags()


class TestParseTagsFromFilename:
    def test_reads_the_bracketed_list(self):
        assert parse_tags_from_filename(
            "['physical design', 'side angle', 'top'].jpg"
        ) == ["physical design", "side angle", "top"]

    def test_ignores_directories_in_the_path(self):
        assert parse_tags_from_filename(
            "/data/train/Mobile/['physical design'].jpg") == ["physical design"]

    def test_normalises_what_it_finds(self):
        assert parse_tags_from_filename("['Physical  Design'].jpg") == [
            "physical design"]

    def test_returns_empty_for_unlabelled_names(self):
        assert parse_tags_from_filename(
            "amazon_co_uk_B09XBQBXS2_Main_Carousel_1.jpg") == []

    def test_returns_empty_rather_than_raising_on_malformed(self):
        assert parse_tags_from_filename("['unclosed.jpg") == []
        assert parse_tags_from_filename("[not, python].jpg") == []


class TestLoadLabelled:
    def test_finds_every_labelled_image(self, labelled_dir):
        pairs = load_labelled(labelled_dir)
        assert len(pairs) == 3
        assert all(tags for _path, tags in pairs)

    def test_all_labels_are_in_the_taxonomy(self, labelled_dir):
        vocab = allowed_tags()
        for _path, tags in load_labelled(labelled_dir):
            assert set(tags) <= vocab


def test_taxonomy_json_is_two_level():
    """Every Specific must sit under a General, so prompts render correctly."""
    assert TAXONOMY
    for general, specifics in TAXONOMY.items():
        assert isinstance(specifics, list) and specifics
        assert normalise(general) == general
