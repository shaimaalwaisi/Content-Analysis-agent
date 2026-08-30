"""Image encoding and response parsing -- the provider boundary."""
import base64
import io

from PIL import Image

from agent.vlm import (MAX_IMAGE_DIM, MockVLM,
                                        OPENAI_COMPATIBLE, PROVIDERS,
                                        encode_image, get_client,
                                        parse_reasons, parse_tag_array,
                                        sniff_media_type)


def _decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class TestSniffMediaType:
    def test_reads_the_bytes_not_the_name(self, png_named_jpg):
        # The training set contains a PNG called .jpg. Declaring image/jpeg
        # for it makes the Anthropic API reject the request with a 400.
        _b64, media = encode_image(png_named_jpg)
        assert media == "image/png"

    def test_known_signatures(self):
        assert sniff_media_type(b"\x89PNG\r\n\x1a\n") == "image/png"
        assert sniff_media_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
        assert sniff_media_type(b"RIFF____WEBP") == "image/webp"
        assert sniff_media_type(b"GIF89a") == "image/gif"

    def test_unknown_falls_back_to_jpeg(self):
        assert sniff_media_type(b"not an image") == "image/jpeg"

    def test_riff_that_is_not_webp_is_not_claimed(self):
        assert sniff_media_type(b"RIFF____WAVE") == "image/jpeg"


class TestEncodeImage:
    def test_small_images_pass_through_untouched(self, jpeg):
        b64, media = encode_image(jpeg)
        assert media == "image/jpeg"
        assert _decode(b64).size == (64, 64)

    def test_large_images_are_downscaled(self, big_jpeg):
        b64, media = encode_image(big_jpeg)
        assert max(_decode(b64).size) == MAX_IMAGE_DIM
        assert media == "image/jpeg"

    def test_downscaling_preserves_aspect_ratio(self, big_jpeg):
        width, height = _decode(encode_image(big_jpeg)[0]).size
        assert round(width / height, 2) == round(2000 / 1500, 2)

    def test_max_dim_zero_sends_the_original(self, big_jpeg):
        assert _decode(encode_image(big_jpeg, max_dim=0)[0]).size == (2000, 1500)


class TestParseTagArray:
    def test_plain_json_array(self):
        assert parse_tag_array('["physical design", "top"]') == [
            "physical design", "top"]

    def test_strips_markdown_fences(self):
        assert parse_tag_array('```json\n["colour"]\n```') == ["colour"]

    def test_finds_the_array_amid_prose(self):
        assert parse_tag_array('Sure! ["camera"] hope that helps') == ["camera"]

    def test_empty_and_malformed_return_empty(self):
        assert parse_tag_array("") == []
        assert parse_tag_array("no array here") == []
        assert parse_tag_array("[unclosed") == []

    def test_coerces_scalar_elements_to_strings(self):
        assert parse_tag_array('["ok", 5]') == ["ok", "5"]

    def test_drops_object_elements(self):
        assert parse_tag_array('["ok", {"a": 1}]') == ["ok"]

    def test_nested_arrays_yield_nothing(self):
        # The extractor takes the first bracketed span, so a nested array
        # produces a malformed slice. Returning [] is the safe outcome:
        # a partial parse would silently invent a tag list.
        assert parse_tag_array('["ok", ["nested"]]') == []


class TestGetClient:
    def test_mock_needs_no_key(self):
        assert isinstance(get_client("mock"), MockVLM)

    def test_openai_compatible_providers_carry_a_base_url(self, monkeypatch):
        for name, cfg in OPENAI_COMPATIBLE.items():
            if name == "openai":
                continue                      # OpenAI itself uses the default
            monkeypatch.setenv(cfg["key_env"], "test-key")
            client = get_client(name)
            assert cfg["base_url"] in str(client._client.base_url)

    def test_missing_key_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        try:
            get_client("xai")
        except RuntimeError as exc:
            assert "XAI_API_KEY" in str(exc)
        else:
            raise AssertionError("expected a RuntimeError naming the variable")

    def test_unknown_provider_lists_the_valid_ones(self):
        try:
            get_client("gemini")
        except ValueError as exc:
            assert "anthropic" in str(exc) and "mock" in str(exc)
        else:
            raise AssertionError("expected a ValueError")

    def test_providers_list_matches_the_factory(self):
        assert set(PROVIDERS) == {"anthropic", "mock", *OPENAI_COMPATIBLE}


class TestParseReasons:
    """Step 1 of a reasoned answer is free text, so parsing it is best-effort
    by design: a missing reason must never cost a tag."""

    ANSWER = (
        "front angle - the TV is photographed face on\n"
        "colour: three colourways are shown\n"
        '["front angle", "colour"]')

    def test_reads_a_reason_per_tag(self):
        assert parse_reasons(self.ANSWER) == {
            "front angle": "the TV is photographed face on",
            "colour": "three colourways are shown"}

    def test_the_answer_array_still_parses_alongside_reasons(self):
        assert parse_tag_array(self.ANSWER) == ["front angle", "colour"]

    def test_accepts_bulleted_and_numbered_lines(self):
        reasons = parse_reasons("- camera - a lens is visible\n"
                                "2. colour - two finishes shown")
        assert reasons == {"camera": "a lens is visible",
                           "colour": "two finishes shown"}

    def test_reads_a_general_and_its_specific_from_one_line(self):
        # What Haiku actually writes when a Specific needs its General.
        reasons = parse_reasons(
            "feature graphics: camera - a ZEISS lens is called out")
        assert reasons["camera"] == "a ZEISS lens is called out"
        assert reasons["feature graphics"].startswith("camera")

    def test_a_dash_inside_a_reason_does_not_invent_a_tag(self):
        reasons = parse_reasons(
            "colour - three finishes - black, white, green - are shown")
        assert list(reasons) == ["colour"], "only real tags may be nested"

    def test_a_plain_answer_has_no_reasons(self):
        assert parse_reasons('["physical design"]') == {}

    def test_ignores_prose_that_is_not_a_reason_line(self):
        assert parse_reasons("Here is my analysis of the image.") == {}

    def test_survives_an_empty_response(self):
        assert parse_reasons("") == {}


class TestMockPredict:
    def test_the_mock_client_reasons_too(self):
        pred = MockVLM().predict("b64", "image/jpeg", context="Category: TV")
        assert pred.tags == ["physical design", "front angle"]
        assert set(pred.reasons) == set(pred.tags)

    def test_predict_tags_is_unchanged(self):
        assert MockVLM().predict_tags("b64", "image/jpeg",
                                      context="Category: TV") == [
            "physical design", "front angle"]
