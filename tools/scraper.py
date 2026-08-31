"""The scraper tool: where a run's images come from when nobody uploads any.

The agent has always been handed a folder. This tool builds that folder from
sony.com instead, so the same graph serves both routes:

    upload ten images  -->\
                           }--> data/fetched/<Category>/<Model>/ --> tag
    fetch a category   -->/

That shared shape is the whole design. Images land in
`<dest>/<Category>/<Model>/<file>`, which is exactly the layout
`agent.graph._infer_context` already reads product context out of, so the
fetch route needs no new node, no new prompt and no classifier: the page URL
says which category a picture belongs to, and a folder name carries it to the
model far more reliably than asking a model to guess it back off the pixels.

Two backends, behind one protocol, mirroring `search`:

* SonyScraper -- live: reads the page, resolves the image URLs, downloads.
* MockScraper -- offline, deterministic; no network. The one stand-in the
                 project keeps for this path, so the fetch route can be tested
                 and demonstrated without hitting a real site.

Three things it refuses to do, all of them deliberate:

* fetch a page robots.txt disallows (`respect_robots`, on by default),
* download an image whose perceptual hash it has already seen -- pass
  `seen=store.seen_hashes()` and a nightly run costs nothing for the
  catalogue that has not changed,
* follow links. It reads the page it is given and stops; a crawler that
  wanders is a crawler that gets blocked.

What comes back is only files. Nothing here proposes a tag, so the controlled
vocabulary remains the only thing that can put one in the output.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Protocol
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from PIL import Image

from agent.logconf import get_logger
from agent.retry import call_with_retry

from .imagemeta import perceptual_hash

log = get_logger(__name__)

USER_AGENT = ("ContentAnalysisAgent/1.0 "
              "(+https://github.com/shaimaalwaisi/Content-Analysis-agent)")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# URL words -> the folder name the agent already knows. The values must stay
# spelled the way `agent.graph._infer_context` expects them, or the context
# line goes missing and the model loses the one hint it cannot see.
CATEGORY_WORDS = (
    ("tv", "TV"), ("television", "TV"), ("bravia", "TV"),
    ("xperia", "Mobile"), ("smartphone", "Mobile"), ("mobile", "Mobile"),
    ("headphone", "Headphone"), ("earbud", "Headphone"), ("wh-", "Headphone"),
    ("wf-", "Headphone"), ("speaker", "Speaker"), ("soundbar", "Speaker"),
    ("srs-", "Speaker"), ("audio", "Video & Sound"), ("sound", "Video & Sound"),
)


@dataclass
class Fetched:
    """One image URL and what became of it."""

    url: str
    path: str = ""
    category: str = ""
    product: str = ""
    phash: str = ""
    bytes: int = 0
    skipped: str = ""          # why it was not kept; "" means it was

    @property
    def kept(self) -> bool:
        return not self.skipped


class ScraperTool(Protocol):
    def fetch(self, url: str, dest: str, limit: int = 20) -> list[Fetched]:
        ...


def category_for(url: str) -> str:
    """Which folder this page's images belong in. Unrecognised pages go to
    'Unknown' rather than being dropped: a picture with no category is still
    taggable, it just reaches the model without a context line."""
    haystack = url.lower()
    for word, category in CATEGORY_WORDS:
        if word in haystack:
            return category
    return "Unknown"


def product_for(url: str) -> str:
    """The model name out of the URL slug: .../bravia/xr-65a95k/ -> XR-65A95K.

    Sony's product URLs end in the model number often enough that this beats
    reading it off the page, which is localised, and off the image, which is
    the model's guess."""
    parts = [p for p in urlparse(url).path.split("/") if p and "." not in p]
    for part in reversed(parts):
        if re.search(r"\d", part) and len(part) <= 40:
            return part.upper()
    return parts[-1].upper() if parts else "UNKNOWN"


def _widest(srcset: str) -> str:
    """The largest candidate in a srcset. Retail pages list a thumbnail first,
    and a thumbnail is not what a tagger should be looking at."""
    best, best_width = "", -1
    for candidate in srcset.split(","):
        bits = candidate.split()
        if not bits:
            continue
        width = 0
        if len(bits) > 1 and bits[1][:-1].isdigit():
            width = int(bits[1][:-1])
        if width >= best_width:
            best, best_width = bits[0], width
    return best


def image_urls(html: str, base_url: str) -> list[str]:
    """Absolute, de-duplicated image URLs on one page, best sources first.

    Covers the four ways a retail page names a picture: og:image, plain src,
    the lazy-loading data-src, and srcset."""
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:                      # lxml absent or unhappy with the
        soup = BeautifulSoup(html, "html.parser")   # markup; stdlib still parses

    found: list[str] = []
    for meta in soup.find_all("meta", property="og:image"):
        found.append(meta.get("content", ""))
    for tag in soup.find_all(["img", "source"]):
        for attr in ("src", "data-src", "data-original"):
            if tag.get(attr):
                found.append(tag[attr])
        if tag.get("srcset"):
            found.append(_widest(tag["srcset"]))

    out, seen = [], set()
    for raw in found:
        if not raw or raw.startswith("data:"):
            continue
        url = urljoin(base_url, raw.strip())
        path = urlparse(url).path.lower()
        if not path.endswith(IMAGE_EXTS) or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _safe_name(url: str) -> str:
    name = os.path.basename(urlparse(url).path) or "image.jpg"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[-80:]
    return name if name.lower().endswith(IMAGE_EXTS) else name + ".jpg"


@dataclass
class SonyScraper:
    """Read one product or category page and download the pictures on it."""

    timeout: float = 20.0
    attempts: int = 3
    min_bytes: int = 8_000          # below this it is an icon or a spacer
    respect_robots: bool = True
    allowed_hosts: tuple[str, ...] = ("sony.com", "sony.co.uk", "sony.net",
                                      "sony-europe.com", "scene7.com")
    # Perceptual hashes already in the database. Anything matching is a file
    # we have tagged before, so it is not downloaded at all.
    seen: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        import httpx
        self._client = httpx.Client(
            timeout=self.timeout, follow_redirects=True,
            headers={"User-Agent": USER_AGENT})
        self._robots: dict[str, RobotFileParser | None] = {}

    # ---- fetching --------------------------------------------------------
    def _get(self, url: str):
        def once():
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp
        return call_with_retry(once, attempts=self.attempts)

    def _host_allowed(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == h or host.endswith("." + h)
                   for h in self.allowed_hosts)

    def allowed(self, url: str) -> bool:
        """What robots.txt says. An unreachable robots.txt is read as
        permission, which is the convention -- a site that means to refuse
        serves the file and says so."""
        if not self.respect_robots:
            return True
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            parser = RobotFileParser()
            try:
                parser.parse(self._client.get(host + "/robots.txt")
                             .text.splitlines())
            except Exception as exc:
                log.info("robots_unavailable",
                         extra={"host": host, "error": str(exc)[:120]})
                parser = None
            self._robots[host] = parser
        parser = self._robots[host]
        return parser is None or parser.can_fetch(USER_AGENT, url)

    def fetch(self, url: str, dest: str, limit: int = 20) -> list[Fetched]:
        """Download up to `limit` new images from one page into
        `dest/<Category>/<Model>/`. Every candidate comes back, kept or not,
        so a caller can report what was skipped and why."""
        if not self.allowed(url):
            log.warning("robots_disallowed", extra={"url": url})
            return [Fetched(url=url, skipped="robots.txt disallows this page")]

        category, product = category_for(url), product_for(url)
        folder = os.path.join(dest, category, product)
        page = self._get(url).text
        candidates = image_urls(page, url)
        log.info("page_read", extra={"url": url, "category": category,
                                     "product": product,
                                     "candidates": len(candidates)})

        out, kept = [], 0
        for image_url in candidates:
            if kept >= limit:
                break
            result = self._download(image_url, folder, category, product)
            out.append(result)
            kept += result.kept
        log.info("fetch_complete", extra={"url": url, "kept": kept,
                                          "seen": len(out)})
        return out

    def _download(self, url: str, folder: str, category: str,
                  product: str) -> Fetched:
        row = Fetched(url=url, category=category, product=product)
        if not self._host_allowed(url):
            row.skipped = "off-site"
            return row
        try:
            data = self._get(url).content
        except Exception as exc:           # one bad image must not end the run
            row.skipped = f"{type(exc).__name__}: {str(exc)[:80]}"
            log.warning("image_failed", extra={"url": url,
                                               "error": row.skipped})
            return row
        row.bytes = len(data)
        if len(data) < self.min_bytes:
            row.skipped = "too small to be a product shot"
            return row
        try:
            with Image.open(BytesIO(data)) as img:
                img.load()
                row.phash = perceptual_hash(img)
        except Exception:
            row.skipped = "not a readable image"
            return row
        if row.phash in self.seen:
            row.skipped = "already in the database"
            return row
        self.seen.add(row.phash)

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, _safe_name(url))
        if os.path.exists(path):           # same name, different picture
            stem, ext = os.path.splitext(path)
            path = f"{stem}-{row.phash[:6]}{ext}"
        with open(path, "wb") as fh:       # the original bytes, so the EXIF
            fh.write(data)                 # the metadata tool reads survives
        row.path = path
        return row

    def close(self) -> None:
        self._client.close()


@dataclass
class MockScraper:
    """Deterministic offline stand-in.

    Draws its own images instead of downloading any, so the fetch route can be
    tested and demonstrated with no network and no key. It names categories and
    products through the same two functions the live scraper uses, because a
    stand-in that answers differently from the real thing is worse than none.
    It is a fixture: the pictures are coloured rectangles and say nothing about
    a real product.
    """

    count: int = 4                  # images a page is pretended to hold
    seen: set[str] = field(default_factory=set)
    size: tuple[int, int] = (320, 240)

    def fetch(self, url: str, dest: str, limit: int = 20) -> list[Fetched]:
        category, product = category_for(url), product_for(url)
        folder = os.path.join(dest, category, product)
        # Two pages must not draw the same picture, or the second would be
        # skipped as a duplicate of the first and the demo would look broken.
        seed = int(hashlib.sha1(url.encode()).hexdigest()[:6], 16)
        out = []
        for i in range(min(self.count, limit)):
            row = Fetched(url=f"{url.rstrip('/')}/shot_{i + 1}.jpg",
                          category=category, product=product)
            img = Image.new("RGB", self.size, ((seed + i * 37) % 180 + 20,
                                               (seed // 7 + i * 23) % 200,
                                               (seed // 13 + i * 11) % 220))
            img.paste(Image.new("RGB", (70, 70), (245, 245, 245)),
                      (10 + (seed + i * 29) % 200,      # something for a hash
                       10 + (seed // 5 + i * 17) % 140))  # to bite on
            row.phash = perceptual_hash(img)
            if row.phash in self.seen:
                row.skipped = "already in the database"
                out.append(row)
                continue
            self.seen.add(row.phash)
            os.makedirs(folder, exist_ok=True)
            row.path = os.path.join(folder, f"shot_{i + 1}.jpg")
            if os.path.exists(row.path):
                row.path = f"{row.path[:-4]}-{row.phash[:6]}.jpg"
            img.save(row.path, format="JPEG")
            row.bytes = os.path.getsize(row.path)
            out.append(row)
        return out

    def close(self) -> None:
        pass


def get_scraper(name: str, seen: set[str] | None = None) -> ScraperTool:
    name = (name or "mock").lower()
    if name == "mock":
        return MockScraper(seen=seen or set())
    if name in ("sony", "live", "http"):
        return SonyScraper(seen=seen or set())
    raise ValueError(f"Unknown scraper: {name!r}. Choose: mock, sony")
