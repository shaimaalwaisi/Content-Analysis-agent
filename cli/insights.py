"""`insights` -- join tags to the metadata sheet and rank them by engagement."""
from __future__ import annotations

import json

from content_analysis_agent.pipeline import results_to_dicts, run_folder
from content_analysis_agent.vlm import get_client

from .common import add_memory_args, add_provider_args, build_memory


def _numeric(value) -> float | None:
    try:
        num = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return None if num != num else num          # drop NaN


def _tagged_records(args) -> list[dict] | None:
    """Where the tags come from: the sheet itself, a saved run, or a fresh one."""
    if args.from_sheet:
        return None
    if args.results:
        with open(args.results) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} tagged results from {args.results}")
        return records
    client = get_client(args.provider, args.model)
    results = run_folder(args.input, client, limit=args.limit,
                         memory=build_memory(args), workers=args.workers)
    records = results_to_dicts(results)
    print(f"Tagged {len(records)} images from {args.input}")
    return records


def run(args) -> None:
    from analysis import (format_engagement, join_tags, load_metadata,
                          rows_from_sheet, tag_engagement,
                          write_synthetic_metadata)

    records = _tagged_records(args)

    path = args.metadata
    if args.synthetic and records is not None:
        path = write_synthetic_metadata([r["path"] for r in records],
                                        args.synthetic)
        print(f"\n*** Using SYNTHETIC metadata written to {path} -- these "
              f"numbers exercise the join, they are not real findings. ***")

    metadata = load_metadata(path)
    if args.from_sheet:
        joined = rows_from_sheet(metadata)
        print(f"Read {len(metadata)} rows from {path}; "
              f"{len(joined)} carry tags in their file name")
    else:
        joined = join_tags(records, metadata)
        print(f"Matched {len(joined)}/{len(records)} tagged images to "
              f"metadata rows")
    if not joined:
        print("Nothing to analyse. The supplied sheet covers the labelled "
              "training images, not the test set -- try --from-sheet.")
        return

    values = [v for v in (_numeric(r.get(args.metric)) for r in joined)
              if v is not None]
    print(f"{len(values)}/{len(joined)} of those have a '{args.metric}' value")
    overall = sum(values) / len(values) if values else None

    report = tag_engagement(joined, metric=args.metric,
                            min_support=args.min_support)
    print(f"\nWhich tags earn attention (ranked by lift on '{args.metric}')")
    print(format_engagement(report, args.metric, overall))

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metric": args.metric, "overall_mean": overall,
                       "synthetic": bool(args.synthetic),
                       "matched": len(joined), "tags": report}, f, indent=2)
        print(f"\nWrote report to {args.output}")


def add_parser(sub) -> None:
    p = sub.add_parser("insights",
                       help="join tags to the metadata sheet and rank tags "
                            "by engagement")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="folder of images to tag, then analyse")
    src.add_argument("--results", help="reuse a results.json from `tag`")
    src.add_argument("--from-sheet", action="store_true",
                     help="take tags from the sheet's own labelled file names "
                          "(no model call; the supplied sheet covers the "
                          "labelled training images)")
    p.add_argument("--metadata", default="data/meta_data.xlsx",
                   help="metadata sheet (.xlsx or .csv)")
    p.add_argument("--synthetic", metavar="PATH", default=None,
                   help="generate a SYNTHETIC sheet at PATH and use it "
                        "(for demos when the real sheet is unavailable)")
    p.add_argument("--metric", default="views",
                   help="numeric column to rank by (default: views)")
    p.add_argument("--min-support", type=int, default=2,
                   help="ignore tags on fewer than N images")
    p.add_argument("--limit", type=int, default=None, help="max images")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="tag N images in parallel")
    p.add_argument("--output", default=None, help="write report JSON")
    add_provider_args(p)
    add_memory_args(p)
    p.set_defaults(func=run)
