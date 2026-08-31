"""`fetch` -- build the folder the agent tags, from sony.com.

`tag --input` has always needed a folder somebody had already assembled. This
command makes one: it reads the product pages you name, downloads the pictures
on them into `<dest>/<Category>/<Model>/`, and records what each file is in the
results database. `--tag` then hands that same folder to a tagging run, which
is the whole route -- fetch, describe, store, tag -- in one line.

Nothing here classifies anything. The page URL says which category its images
belong to, and the folder name carries that to the model; asking a model to
read the category back off a photograph would be paying for an answer the URL
already gave. See `tools.scraper.category_for`.

Two things keep a second run cheap: the database's perceptual hashes go in as
the scraper's `seen` set, so a picture already on file is never downloaded
again, and the tag memory means an image that somehow arrives twice is not
sent to the model twice either.
"""
from __future__ import annotations

import os

from . import tag as tag_command
from .common import (add_enrich_args, add_memory_args, add_provider_args,
                     add_results_db_args, build_store)
from .runlog import write_run


def _size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1048576:.1f} MB"


def _skipped_summary(rows) -> str:
    """Skips grouped by reason -- 'skipped 14' says nothing useful, and the
    reasons are how you tell a blocked scraper from a page you already have."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.skipped] = counts.get(row.skipped, 0) + 1
    return ", ".join(f"{n} {reason}" for reason, n in
                     sorted(counts.items(), key=lambda kv: -kv[1]))


def run(args) -> None:
    from tools import get_scraper, read_metadata

    if args.html and len(args.url) != 1:
        raise SystemExit("--html reads one saved page, so give exactly one "
                         "--url: the address that page came from.")

    store = build_store(args)
    seen = store.seen_hashes() if store else set()
    if seen:
        print(f"{len(seen)} picture(s) already on file; those are not "
              f"downloaded again.")
    scraper = get_scraper(args.scraper, seen=seen)

    saved = None
    if args.html:
        with open(args.html, encoding="utf-8", errors="replace") as handle:
            saved = handle.read()
        print(f"Reading {args.html} as the page for {args.url[0]}; only the "
              f"images are downloaded.")

    kept, skipped, records = [], [], []
    try:
        for url in args.url:
            print(f"\n{url}")
            rows = scraper.fetch(url, args.dest, limit=args.per_page,
                                 html=saved)
            page_skipped = [r for r in rows if not r.kept]
            for row in (r for r in rows if r.kept):
                # The metadata is read from the file on disk, so what lands in
                # the database describes the bytes we actually kept.
                meta = read_metadata(row.path, row.category, row.product)
                if store:
                    store.put_image(meta, row.url)
                kept.append(row)
                records.append({**meta.as_dict(), "source_url": row.url})
                print(f"  + {os.path.relpath(row.path, args.dest)}"
                      f"   {meta.width}x{meta.height}   {_size(meta.bytes)}")
            if page_skipped:
                print(f"  - skipped {len(page_skipped)}: "
                      f"{_skipped_summary(page_skipped)}")
            skipped += page_skipped
    finally:
        scraper.close()
        if store:
            store.close()

    print(f"\nFetched {len(kept)} new image(s) from {len(args.url)} page(s) "
          f"into {args.dest}; skipped {len(skipped)}.")
    run_path = write_run("fetch", args, {
        "pages": list(args.url), "fetched": len(kept), "skipped": len(skipped),
        "skipped_reasons": _skipped_summary(skipped) if skipped else "",
        "images": records})
    if run_path:
        print(f"Run record: {run_path}")

    if not kept:
        return                     # the skip reasons above already said why
    if not args.tag:
        print(f"\nTag them with:\n"
              f"  python -m cli tag --input {args.dest} --few-shot 8")
        return
    # Hand the folder to the tagging command itself rather than reassembling
    # the run here: one code path decides how a folder gets tagged.
    print(f"\nTagging {args.dest} ...")
    args.input, args.limit = args.dest, None
    tag_command.run(args)


def add_parser(sub) -> None:
    p = sub.add_parser("fetch",
                       help="download product images from sony.com and record "
                            "them")
    # extend, so both `--url A B` and a repeated `--url` accumulate rather
    # than the last one silently winning.
    p.add_argument("--url", nargs="+", action="extend", required=True,
                   metavar="URL", help="product or category page(s) to read")
    p.add_argument("--html", default=None, metavar="FILE",
                   help="a page saved from your browser to read instead of "
                        "fetching it, for sites that refuse a script. Needs "
                        "exactly one --url, which says which page it is.")
    p.add_argument("--dest", default="data/fetched",
                   help="where the Category/Model folders are written")
    p.add_argument("--per-page", type=int, default=20, metavar="N",
                   help="keep at most N new images per page")
    p.add_argument("--scraper", default="sony", choices=["sony", "mock"],
                   help="'mock' draws its own images: no network, for trying "
                        "the route out")
    p.add_argument("--tag", action="store_true",
                   help="tag what was fetched straight away, with the flags "
                        "below")
    # The tagging flags, so --tag is a real tagging run rather than a
    # stripped-down copy of one. --limit and --input are set from --per-page
    # and --dest, so they are not offered twice.
    p.add_argument("--output", default=None, help="results .json or .csv")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="tag N images in parallel (network-bound work)")
    p.add_argument("--few-shot", type=int, default=0, metavar="N",
                   help="prepend N labelled training images as examples")
    p.add_argument("--train-dir", default="data/train",
                   help="where --few-shot examples come from")
    add_provider_args(p)
    add_memory_args(p)
    add_enrich_args(p)
    add_results_db_args(p)
    p.set_defaults(func=run, consistency=False)
