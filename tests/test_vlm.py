"""Image encoding and response parsing -- the provider boundary."""
import base64
import io

import pytest
from PIL import Image

from agent.vlm import (DEFAULT_MODEL, MAX_IMAGE_DIM, PROVIDERS, Prediction,
                       encode_image, get_client, parse_details, parse_reasons,
                       parse_tag_array, sniff_media_type)


def _decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class TestEncodeImage:
    def test_media_type_comes_from_the_bytes_not_the_name(self, png_named_jpg):
        # The training set contains a PNG called .jpg. Declaring image/jpeg
        # for it makes the Anthropic API reject the request with a 400.
        assert encode_image(png_named_jpg)[1] == "image/png"
        assert sniff_media_type(b"\x89PNG\r\n\x1a\n") == "image/png"
        assert sniff_media_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
        assert sniff_media_type(b"RIFF____WEBP") == "image/webp"
        assert sniff_media_type(b"GIF89a") == "image/gif"
        # A RIFF container that is not WEBP, and anything unrecognised, fall
        # back rather than claim a type the bytes do not support.
        assert sniff_media_type(b"RIFF____WAVE") == "image/jpeg"
        assert sniff_media_type(b"not an image") == "image/jpeg"

    def test_small_images_pass_through_untouched(self, jpeg):
        b64, media = encode_image(jpeg)
        assert media == "image/jpeg" and _decode(b64).size == (64, 64)

    def test_large_images_are_downscaled_in_proportion(self, big_jpeg):
        width, height = _decode(encode_image(big_jpeg)[0]).size
        assert max(width, height) == MAX_IMAGE_DIM
        assert round(width / height, 2) == round(2000 / 1500, 2)
        # max_dim=0 is the escape hatch: send exactly what is on disk.
        assert _decode(encode_image(big_jpeg, max_dim=0)[0]).size == (2000, 1500)


class TestParseTagArray:
    def test_finds_the_array_however_it_is_wrapped(self):
        assert parse_tag_array('["physical design", "top"]') == [
            "physical design", "top"]
        assert parse_tag_array('```json\n["colour"]\n```') == ["colour"]
        assert parse_tag_array('Sure! ["camera"] hope that helps') == ["camera"]

    def test_a_malformed_answer_yields_nothing_and_odd_elements_are_fixed(
            self):
        # A nested array yields nothing at all: the extractor takes the first
        # bracketed span, so a partial parse would invent a tag list.
        for text in ("", "no array here", "[unclosed", '["ok", ["nested"]]'):
            assert parse_tag_array(text) == []
        assert parse_tag_array('["ok", 5]') == ["ok", "5"]
        assert parse_tag_array('["ok", {"a": 1}]') == ["ok"]


class TestGetClient:
    def test_anthropic_is_the_only_provider_and_its_model_is_priced(self):
        # The default must be a model evaluation.runstats can price, or cost
        # per task silently reports "unpriced" on every real run.
        from evaluation.runstats import price_for
        assert PROVIDERS == ["anthropic"]
        assert price_for(DEFAULT_MODEL) is not None

    def test_a_removed_provider_says_so(self):
        for gone in ("mock", "openai", "xai", "groq", "ollama", "gemini"):
            with pytest.raises(ValueError, match="Claude only"):
                get_client(gone)


class TestParseReasons:
    """Step 1 of a reasoned answer is free text, so parsing it is best-effort
    by design: a missing reason must never cost a tag."""

    ANSWER = (
        "front angle - the TV is photographed face on\n"
        "colour: three colourways are shown\n"
        '["front angle", "colour"]')

    def test_reads_a_reason_per_tag_without_disturbing_the_array(self):
        assert parse_reasons(self.ANSWER) == {
            "front angle": "the TV is photographed face on",
            "colour": "three colourways are shown"}
        assert parse_tag_array(self.ANSWER) == ["front angle", "colour"]

    def test_accepts_the_shapes_the_model_actually_writes(self):
        assert parse_reasons("- camera - a lens is visible\n"
                             "2. colour - two finishes shown") == {
            "camera": "a lens is visible", "colour": "two finishes shown"}
        # A Specific written alongside its General, on one line.
        reasons = parse_reasons(
            "feature graphics: camera - a ZEISS lens is called out")
        assert reasons["camera"] == "a ZEISS lens is called out"
        assert reasons["feature graphics"].startswith("camera")

    def test_a_dash_inside_a_reason_does_not_invent_a_tag(self):
        reasons = parse_reasons(
            "colour - three finishes - black, white, green - are shown")
        assert list(reasons) == ["colour"], "only real tags may be nested"

    def test_an_answer_with_no_reasons_yields_none(self):
        for text in ('["physical design"]', "Here is my analysis.", ""):
            assert parse_reasons(text) == {}


REASONED_ANSWER = (
    "front angle - the TV is photographed face on\n"
    "colour - three finishes are shown\n"
    "Category: TV\n"
    "Model: XR-65A95K\n"
    "Description: A 65-inch OLED television shown face on in three finishes.\n"
    "Specs: 65-inch, OLED, 4K, XR Processor\n"
    '["front angle", "colour"]')


class TestParseDetails:
    """The four facts the tag vocabulary cannot express, read from the same
    answer as the tags."""

    def test_a_whole_answer_parses_into_one_prediction(self):
        pred = Prediction.from_text(REASONED_ANSWER)
        assert pred.tags == ["front angle", "colour"]
        assert parse_tag_array(REASONED_ANSWER) == pred.tags
        assert pred.product == "XR-65A95K" and pred.category == "TV"
        assert pred.description.startswith("A 65-inch OLED")
        assert pred.specs == "65-inch, OLED, 4K, XR Processor"
        assert set(pred.reasons) == {"front angle", "colour"}, \
            "a detail line is a fact about the product, not a tag reason"

    def test_unknown_and_none_are_left_empty(self):
        # A cell reading "unknown" is worse than an empty one.
        details = parse_details("Model: unknown\nSpecs: none\nCategory: TV")
        assert "model" not in details and "specs" not in details
        assert details["category"] == "TV"

    def test_survives_a_bolded_label_and_ignores_a_plain_answer(self):
        assert parse_details("**Category:** Mobile")["category"] == "Mobile"
        assert parse_details('["physical design"]') == {}
