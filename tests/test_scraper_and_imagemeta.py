"""The fetch route: reading a page, keeping the right images, and what a file
says about itself.

No network anywhere -- the live scraper is exercised through its pure parts
(URL resolution, category and product naming) and the download path through
MockScraper, exactly as the search tool is tested through MockSearchTool.
"""
import os

import pytest
from PIL import Image

from tools import (ImageMeta, MockScraper, ResultStore, category_for,
                   get_scraper, hamming, image_urls, perceptual_hash,
                   product_for, read_metadata)
from tools.scraper import _safe_name, _widest


class TestImageUrls:
    HTML = """
      <meta property="og:image" content="/media/hero.jpg">
      <img src="//cdn.sony.com/front.png">
      <img data-src="/lazy/side.webp" src="data:image/gif;base64,AAAA">
      <img src="/icons/cart.svg">
      <img src="/media/hero.jpg">
      <source srcset="/s-300.jpg 300w, /s-1200.jpg 1200w">
    """
    BASE = "https://www.sony.co.uk/bravia/xr-65a95k/"

    def test_resolves_every_way_a_page_names_a_picture(self):
        urls = image_urls(self.HTML, self.BASE)
        assert urls[0] == "https://www.sony.co.uk/media/hero.jpg", "og first"
        assert "https://cdn.sony.com/front.png" in urls, "protocol-relative"
        assert "https://www.sony.co.uk/lazy/side.webp" in urls, "lazy loaded"

    def test_drops_what_is_not_a_product_photo(self):
        urls = image_urls(self.HTML, self.BASE)
        assert not any(u.endswith(".svg") for u in urls), "icons are not shots"
        assert not any(u.startswith("data:") for u in urls)
        assert len(urls) == len(set(urls)), "the same file listed twice is one"

    def test_srcset_takes_the_largest_candidate(self):
        # A thumbnail is not what a tagger should be looking at.
        assert _widest("/s-300.jpg 300w, /s-1200.jpg 1200w") == "/s-1200.jpg"
        assert "https://www.sony.co.uk/s-1200.jpg" in image_urls(self.HTML,
                                                                 self.BASE)
        assert _widest("") == ""


class TestNaming:
    def test_category_comes_off_the_url(self):
        assert category_for("https://sony.co.uk/electronics/televisions/x") == "TV"
        assert category_for("https://sony.co.uk/xperia-10-v") == "Mobile"
        assert category_for("https://sony.co.uk/headphones/wh-1000xm5") == "Headphone"
        # An unrecognised page is still taggable, just without a context line.
        assert category_for("https://sony.co.uk/about") == "Unknown"

    def test_product_is_the_model_bearing_slug(self):
        assert product_for("https://sony.co.uk/bravia/xr-65a95k/") == "XR-65A95K"
        assert product_for("https://sony.co.uk/tv/xr-65a95k/gallery.html") \
            == "XR-65A95K"
        assert product_for("https://sony.co.uk/") == "UNKNOWN"

    def test_file_names_are_safe_and_keep_an_extension(self):
        assert _safe_name("https://x.com/a/hero shot.jpg") == "hero_shot.jpg"
        assert _safe_name("https://x.com/a/../evil").endswith(".jpg")
        assert "/" not in _safe_name("https://x.com/a/b/c.png")


class TestMockScraper:
    def test_writes_the_category_model_layout_the_agent_reads(self, tmp_path):
        rows = MockScraper().fetch("https://sony.com/bravia/xr-65a95k", str(tmp_path))
        assert all(r.kept for r in rows)
        assert rows[0].path.endswith(
            os.path.join("TV", "XR-65A95K", "shot_1.jpg"))
        # ...which is the layout _infer_context reads product context out of.
        assert read_metadata(rows[0].path).category == "TV"

    def test_limit_is_a_limit_on_images_kept(self, tmp_path):
        assert len(MockScraper().fetch("https://sony.com/bravia/xr-65a95k",
                                       str(tmp_path), limit=2)) == 2

    def test_a_hash_already_on_file_is_not_downloaded_again(self, tmp_path):
        scraper = MockScraper()
        first = scraper.fetch("https://sony.com/bravia/xr-65a95k", str(tmp_path / "a"))
        again = scraper.fetch("https://sony.com/bravia/xr-65a95k", str(tmp_path / "b"))
        assert [r.skipped for r in again] == ["already in the database"] * 4
        assert not (tmp_path / "b").exists(), "nothing was written"
        assert all(r.kept for r in first)

    def test_the_seen_set_can_come_from_the_database(self, tmp_path):
        store = ResultStore(str(tmp_path / "r.sqlite3"))
        rows = MockScraper().fetch("https://sony.com/bravia/xr-65a95k", str(tmp_path))
        for row in rows:
            store.put_image(read_metadata(row.path), row.url)
        fresh = get_scraper("mock", seen=store.seen_hashes())
        assert all(not r.kept for r in fresh.fetch("https://sony.com/bravia/xr-65a95k",
                                                   str(tmp_path / "again")))
        store.close()

    def test_only_the_two_backends_are_accepted(self):
        assert isinstance(get_scraper("mock"), MockScraper)
        with pytest.raises(ValueError, match="Unknown scraper"):
            get_scraper("selenium")


class TestReadMetadata:
    def test_reads_the_file_not_the_model(self, tmp_path):
        folder = tmp_path / "TV" / "XR-65A95K"
        folder.mkdir(parents=True)
        path = str(folder / "hero.jpg")
        Image.new("RGB", (300, 200), (200, 30, 30)).save(path)

        meta = read_metadata(path)
        assert isinstance(meta, ImageMeta)
        assert (meta.width, meta.height, meta.fmt) == (300, 200, "JPEG")
        assert meta.mime == "image/jpeg" and meta.bytes > 0
        assert meta.megapixels == 0.06
        assert len(meta.sha256) == 64 and len(meta.phash) == 16
        # The folder names the product, so a fetched and an uploaded image
        # describe themselves the same way.
        assert (meta.category, meta.product) == ("TV", "XR-65A95K")

    def test_a_png_named_jpg_reports_what_it_actually_is(self, png_named_jpg):
        # The training set contains one of these.
        meta = read_metadata(png_named_jpg)
        assert meta.fmt == "PNG" and meta.mime == "image/png"

    def test_caller_supplied_context_wins_over_the_path(self, jpeg):
        meta = read_metadata(jpeg, category="Mobile", product="XPERIA")
        assert (meta.category, meta.product) == ("Mobile", "XPERIA")


class TestPerceptualHash:
    def _shot(self, tmp_path, name, size=(400, 300)):
        """The same composition at whatever size is asked for: a pale panel
        on a dark ground, always over the same tenth of the frame."""
        w, h = size
        img = Image.new("RGB", size, (20, 90, 160))
        img.paste(Image.new("RGB", (w // 3, h // 3), (240, 240, 240)),
                  (w // 10, h // 10))
        path = str(tmp_path / name)
        img.save(path)
        return path

    def test_the_same_shot_resized_hashes_the_same(self, tmp_path):
        big = self._shot(tmp_path, "big.jpg", (800, 600))
        small = self._shot(tmp_path, "small.jpg", (400, 300))
        # This is the case sha256 gets wrong: one picture, two files.
        assert hamming(perceptual_hash(big), perceptual_hash(small)) <= 5

    def test_a_different_picture_hashes_differently(self, tmp_path):
        shot = self._shot(tmp_path, "a.jpg")
        other = str(tmp_path / "b.jpg")
        Image.new("RGB", (400, 300), (250, 250, 250)).save(other)
        assert hamming(perceptual_hash(shot), perceptual_hash(other)) > 5

    def test_hashes_of_different_lengths_are_a_programming_error(self):
        with pytest.raises(ValueError, match="different lengths"):
            hamming("ff", "ffff")


class TestImagesTable:
    def test_rows_are_keyed_by_picture_not_by_file(self, tmp_path):
        store = ResultStore(str(tmp_path / "r.sqlite3"))
        rows = MockScraper().fetch("https://sony.com/bravia/xr-65a95k", str(tmp_path))
        for row in rows:
            store.put_image(read_metadata(row.path), row.url)
        assert len(store.seen_hashes()) == 4
        # Re-fetching the same picture updates its row rather than adding one.
        store.put_image(read_metadata(rows[0].path), rows[0].url)
        assert len(store.images()) == 4
        assert store.images(category="TV")[0]["source_url"].startswith("https")
        assert store.images(category="Mobile") == []
        assert len(store.images(limit=2)) == 2
        store.close()

    def test_it_does_not_disturb_the_taggings_table(self, tmp_path):
        from tools import Tagging, new_run_id
        store = ResultStore(str(tmp_path / "r.sqlite3"))
        store.put(Tagging(run_id=new_run_id(), image_path="/a/b.jpg",
                          tags=["colour"]))
        store.put_image(read_metadata(
            MockScraper().fetch("https://sony.com/bravia/xr-65a95k",
                                str(tmp_path))[0].path))
        assert len(store.rows()) == 1 and len(store.images()) == 1
        store.close()


class TestFetchCommand:
    """`fetch` end to end on the mock backend: no network, no key, no model."""

    URL = "https://sony.com/bravia/xr-65a95k"

    def _args(self, tmp_path, *extra):
        from cli.main import build_parser
        return build_parser().parse_args(
            ["--no-results", "fetch", "--url", self.URL,
             "--scraper", "mock", "--dest", str(tmp_path / "fetched"),
             "--db", str(tmp_path / "r.sqlite3"), *extra])

    def test_downloads_records_and_prints_what_arrived(self, tmp_path, capsys):
        from cli import fetch

        args = self._args(tmp_path)
        fetch.run(args)
        out = capsys.readouterr().out

        files = sorted(str(p) for p in
                       (tmp_path / "fetched").rglob("*.jpg"))
        assert len(files) == 4
        assert os.path.join("TV", "XR-65A95K") in files[0], "the URL's folder"
        assert "Fetched 4 new image(s)" in out
        assert "320x240" in out, "the metadata tool ran"
        assert "python -m cli tag --input" in out, "says what to do next"

        store = ResultStore(str(tmp_path / "r.sqlite3"))
        assert len(store.seen_hashes()) == 4
        assert store.images()[0]["source_url"].startswith(self.URL)
        store.close()

    def test_a_second_run_downloads_nothing_and_says_so(self, tmp_path,
                                                        capsys):
        from cli import fetch

        fetch.run(self._args(tmp_path))
        capsys.readouterr()
        fetch.run(self._args(tmp_path))
        out = capsys.readouterr().out

        assert "4 picture(s) already on file" in out
        assert "skipped 4: 4 already in the database" in out
        assert "Fetched 0 new image(s)" in out
        assert len(list((tmp_path / "fetched").rglob("*.jpg"))) == 4

    def test_repeated_url_flags_accumulate(self, tmp_path):
        args = self._args(tmp_path, "--url", "https://sony.com/xperia-10-v")
        # nargs='+' alone would let the second --url silently replace the first.
        assert args.url == [self.URL, "https://sony.com/xperia-10-v"]

    def test_two_pages_land_in_their_own_folders(self, tmp_path, capsys):
        from cli import fetch

        fetch.run(self._args(tmp_path, "--url", "https://sony.com/xperia-10-v",
                             "--per-page", "2"))
        folders = {str(p.relative_to(tmp_path / "fetched").parent)
                   for p in (tmp_path / "fetched").rglob("*.jpg")}
        assert folders == {os.path.join("TV", "XR-65A95K"),
                           os.path.join("Mobile", "XPERIA-10-V")}
        assert "Fetched 4 new image(s) from 2 page(s)" in capsys.readouterr().out

    def test_tag_hands_the_fetched_folder_to_the_tagging_command(
            self, tmp_path, monkeypatch):
        from cli import fetch

        seen = {}
        monkeypatch.setattr(fetch.tag_command, "run",
                            lambda args: seen.update(vars(args)))
        fetch.run(self._args(tmp_path, "--tag", "--few-shot", "3"))
        # --tag must run a real tagging run, not a stripped-down copy of one:
        # the folder just fetched, and the tag flags as given.
        assert seen["input"] == str(tmp_path / "fetched")
        assert seen["limit"] is None, "--per-page is not a tagging limit"
        assert seen["few_shot"] == 3 and seen["provider"] == "anthropic"

    def test_no_db_still_fetches_it_just_remembers_nothing(self, tmp_path,
                                                           capsys):
        from cli import fetch

        args = self._args(tmp_path, "--no-db")
        fetch.run(args)
        fetch.run(args)
        assert "Fetched 4 new image(s)" in capsys.readouterr().out
        assert not (tmp_path / "r.sqlite3").exists()
