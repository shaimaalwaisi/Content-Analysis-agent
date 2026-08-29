"""The metadata join and engagement lift, and the search tool layer."""
import pandas as pd
import pytest

from analysis.metadata import (_canonical, join_tags, load_metadata,
                               rows_from_sheet, tag_engagement,
                               write_synthetic_metadata)
from agent.evidence import EVIDENCE_RULES, tags_from_evidence
from agent.taxonomy import allowed_tags
from agent.tools_types import SearchResult
from tools import MockSearchTool, get_search_tool


class TestColumnMapping:
    def test_the_supplied_sheets_headers(self):
        # The real meta_data.xlsx calls the file column plain 'Name', and
        # 'Product price' must resolve to price rather than product.
        mapping = _canonical(["Name", "Category", "Model", "Image views",
                              "Product price"])
        assert mapping == {"Name": "file", "Category": "category",
                           "Model": "product", "Image views": "views",
                           "Product price": "price"}

    def test_other_plausible_phrasings(self):
        mapping = _canonical(["File Name", "Product Name", "Category",
                              "Price (GBP)", "Image Views"])
        assert mapping["File Name"] == "file"
        assert mapping["Product Name"] == "product"
        assert mapping["Image Views"] == "views"

    def test_snake_case_variants(self):
        mapping = _canonical(["image_filename", "product_category", "views_30d"])
        assert mapping == {"image_filename": "file",
                           "product_category": "category",
                           "views_30d": "views"}

    def test_unrecognised_headers_are_ignored(self):
        assert _canonical(["random", "columns"]) == {}


class TestLoadMetadata:
    def test_reads_xlsx(self, tmp_path):
        path = tmp_path / "m.xlsx"
        pd.DataFrame({"Name": ["a.jpg"], "Image views": [100]}).to_excel(
            path, index=False)
        records = load_metadata(str(path))
        assert records[0]["file"] == "a.jpg" and records[0]["views"] == 100

    def test_reads_csv(self, tmp_path):
        path = tmp_path / "m.csv"
        pd.DataFrame({"Name": ["a.jpg"], "Image views": [100]}).to_csv(
            path, index=False)
        assert load_metadata(str(path))[0]["file"] == "a.jpg"

    def test_missing_file_names_the_flag(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--metadata"):
            load_metadata(str(tmp_path / "absent.xlsx"))

    def test_a_sheet_with_no_file_column_is_rejected(self, tmp_path):
        path = tmp_path / "m.csv"
        pd.DataFrame({"random": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="file-name column"):
            load_metadata(str(path))


class TestJoinAndSheetTags:
    METADATA = [{"file": "a.jpg", "category": "TV", "product": "X", "views": 10},
                {"file": "b.jpg", "category": "TV", "product": "Y", "views": 20}]

    def test_matches_on_file_name(self):
        joined = join_tags([{"path": "/img/a.jpg", "tags": ["colour"]}],
                           self.METADATA)
        assert len(joined) == 1 and joined[0]["views"] == 10

    def test_unmatched_rows_are_dropped(self):
        assert join_tags([{"path": "/img/zz.jpg", "tags": []}],
                         self.METADATA) == []

    def test_rows_from_sheet_reads_tags_out_of_the_names(self):
        rows = rows_from_sheet([
            {"file": "['physical design', 'top'].jpg", "views": 5},
            {"file": "unlabelled.jpg", "views": 5}])
        assert len(rows) == 1
        assert rows[0]["tags"] == ["physical design", "top"]


class TestTagEngagement:
    JOINED = [{"tags": ["hero"], "views": 300},
              {"tags": ["hero"], "views": 500},
              {"tags": ["dull"], "views": 100},
              {"tags": ["dull"], "views": 100}]

    def test_lift_is_the_mean_over_the_overall_mean(self):
        # overall = 250; hero mean 400 -> 1.6; dull mean 100 -> 0.4
        report = {r["tag"]: r for r in tag_engagement(self.JOINED, "views")}
        assert report["hero"]["mean_views"] == 400.0
        assert report["hero"]["lift"] == pytest.approx(1.6)
        assert report["dull"]["lift"] == pytest.approx(0.4)

    def test_sorted_by_lift(self):
        assert [r["tag"] for r in tag_engagement(self.JOINED, "views")] == [
            "hero", "dull"]

    def test_min_support_hides_thin_tags(self):
        rows = self.JOINED + [{"tags": ["rare"], "views": 9999}]
        tags = [r["tag"] for r in tag_engagement(rows, "views", min_support=2)]
        assert "rare" not in tags

    def test_non_numeric_values_are_skipped_not_fatal(self):
        rows = self.JOINED + [{"tags": ["hero"], "views": "n/a"}]
        report = {r["tag"]: r for r in tag_engagement(rows, "views")}
        assert report["hero"]["mean_views"] == 400.0

    def test_no_metric_yields_an_empty_report(self):
        assert tag_engagement([{"tags": ["hero"]}], "views") == []


class TestSyntheticSheet:
    def test_round_trips_through_load_metadata(self, tmp_path):
        images = [str(tmp_path / "TV" / "MODEL" / "shot_1.jpg")]
        out = write_synthetic_metadata(images, str(tmp_path / "s.csv"))
        records = load_metadata(out)
        assert records[0]["file"] == "shot_1.jpg"
        assert float(records[0]["views"]) > 0

    def test_is_deterministic_for_a_seed(self, tmp_path):
        images = [str(tmp_path / "TV" / "M" / "a_1.jpg")]
        first = load_metadata(write_synthetic_metadata(
            images, str(tmp_path / "a.csv"), seed=1))
        second = load_metadata(write_synthetic_metadata(
            images, str(tmp_path / "b.csv"), seed=1))
        assert first[0]["views"] == second[0]["views"]


class TestEvidenceRules:
    def test_every_enrichable_tag_is_in_the_taxonomy(self):
        # Otherwise enrichment would propose tags validate_tags always drops.
        assert set(EVIDENCE_RULES) <= allowed_tags()

    def test_award_evidence_proposes_the_awards_tag(self):
        assert "awards" in tags_from_evidence(
            [SearchResult("r", "Winner of the EISA best OLED award")])

    def test_matches_on_word_boundaries(self):
        # 'towards' must not fire 'awards'.
        assert tags_from_evidence(
            [SearchResult("r", "a step towards better sound")]) == []

    def test_no_evidence_proposes_nothing(self):
        assert tags_from_evidence([]) == []

    def test_a_tag_is_proposed_at_most_once(self):
        results = [SearchResult("a", "award winner"),
                   SearchResult("b", "another award")]
        assert tags_from_evidence(results).count("awards") == 1


class TestSearchTools:
    def test_mock_needs_no_key(self):
        assert isinstance(get_search_tool("mock"), MockSearchTool)

    def test_mock_returns_nothing_for_an_unknown_product(self):
        assert get_search_tool("mock").search("Sony NOT-A-MODEL") == []

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown search tool"):
            get_search_tool("bing")
