"""Command-line demo for the content tagging agent.

Examples
--------
# See the controlled vocabulary
python -m content_analysis_agent.cli taxonomy

# Tag a folder of test images -> JSON (offline, no key needed)
python -m content_analysis_agent.cli tag --input data/test/TV --provider mock

# Tag with Claude vision and write results
python -m content_analysis_agent.cli tag --input data/test --provider anthropic \
    --output results.json

# Score the agent against the labels baked into the train filenames
python -m content_analysis_agent.cli eval --train-dir data/train \
    --provider anthropic --sample 30 --report metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .logconf import setup_logging
from .evaluation import RunStats
from .tools import get_search_tool
from .memory import DEFAULT_PATH as MEMORY_PATH, TagMemory
from .pipeline import results_to_dicts, run_folder
from .taxonomy import allowed_tags, taxonomy_prompt
from .vlm import PROVIDERS, get_client


def _cmd_taxonomy(_args) -> None:
    tags = sorted(allowed_tags())
    print(taxonomy_prompt())
    print(f"\n{len(tags)} allowed tags total.")


def _cmd_tag(args) -> None:
    client = get_client(args.provider, args.model)

    examples = None
    if args.few_shot:
        from .fewshot import load_examples
        examples = load_examples(args.train_dir, limit=args.few_shot)
        print(f"Using {len(examples)} few-shot example(s) from {args.train_dir}")

    def progress(i, total, res):
        name = res.path.rsplit("/", 1)[-1]
        print(f"[{i}/{total}] {name} -> {res.tags}")

    memory = None if args.no_memory else TagMemory(args.memory)
    stats = RunStats()
    tool = get_search_tool(args.search_tool) if args.enrich else None
    results = run_folder(args.input, client, limit=args.limit,
                         on_item=progress, examples=examples, memory=memory,
                         workers=args.workers, search_tool=tool, stats=stats)
    print(f"\n{stats.summary()}")
    if memory:
        print(memory.summary())
    records = results_to_dicts(results)

    if args.output:
        if args.output.endswith(".csv"):
            with open(args.output, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "category", "model", "tags"])
                for r in records:
                    w.writerow([r["path"], r["category"], r["model"],
                                "; ".join(r["tags"])])
        else:
            with open(args.output, "w") as f:
                json.dump(records, f, indent=2)
        print(f"\nWrote {len(records)} results to {args.output}")
    else:
        print(json.dumps(records, indent=2))


def _cmd_eval(args) -> None:
    # local imports: the mock path needs no key
    from .evaluation import (compare_baselines, evaluate, failure_warning,
                             format_comparison)

    client = get_client(args.provider, args.model)

    def progress(i, total, path, gt, got):
        name = path.rsplit("/", 1)[-1]
        mark = "OK " if set(gt) == set(got) else "xx "
        print(f"[{i}/{total}] {mark}{name}\n      truth={gt}\n      pred ={got}")

    if args.few_shot:
        print(f"Using up to {args.few_shot} few-shot example(s) "
              f"(the scored image is always excluded from its own examples)")
    memory = None if args.no_memory else TagMemory(args.memory)
    stats = RunStats()
    tool = get_search_tool(args.search_tool) if args.enrich else None
    metrics, records = evaluate(args.train_dir, client, sample=args.sample,
                                on_item=progress, few_shot=args.few_shot,
                                memory=memory, search_tool=tool, stats=stats)
    warning = failure_warning(records)
    if warning:
        print(warning)
    print("\n" + "=" * 48)
    print(metrics.summary())
    if memory:
        print(memory.summary())
    print("=" * 48)
    print("\n" + stats.summary())

    scored = {}
    if not args.no_baseline:
        truth = [r["truth"] for r in records]
        pred = [r["predicted"] for r in records]
        scored = compare_baselines(truth, pred)
        print("\nAgent vs model-free baselines")
        print(format_comparison(scored))
        if warning:
            print("\n(Scores above are unreliable - see the failure warning.)")

    if args.report:
        payload = {"metrics": {k: v for k, v in vars(metrics).items()
                               if k != "per_tag"},
                   "workflow": stats.as_dict(),
                   "per_tag": metrics.per_tag,
                   "baselines": {name: {k: v for k, v in vars(m).items()
                                        if k != "per_tag"}
                                 for name, m in scored.items()
                                 if name != "agent"},
                   "records": records}
        with open(args.report, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull report written to {args.report}")


def _cmd_insights(args) -> None:
    from .metadata import (format_engagement, join_tags, load_metadata,
                           rows_from_sheet, tag_engagement,
                           write_synthetic_metadata)

    if args.from_sheet:
        records = None          # tags come from the sheet itself
    elif args.results:
        with open(args.results) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} tagged results from {args.results}")
    else:
        client = get_client(args.provider, args.model)
        memory = None if args.no_memory else TagMemory(args.memory)
        results = run_folder(args.input, client, limit=args.limit,
                             memory=memory, workers=args.workers)
        records = results_to_dicts(results)
        print(f"Tagged {len(records)} images from {args.input}")

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

    with_metric = sum(1 for r in joined
                      if str(r.get(args.metric, "")).strip() not in ("", "nan"))
    print(f"{with_metric}/{len(joined)} of those have a '{args.metric}' value")

    report = tag_engagement(joined, metric=args.metric,
                            min_support=args.min_support)
    vals = [float(str(r.get(args.metric)).replace(",", ""))
            for r in joined if str(r.get(args.metric)).replace(",", "")
            .replace(".", "").isdigit()]
    overall = sum(vals) / len(vals) if vals else None
    print(f"\nWhich tags earn attention (ranked by lift on '{args.metric}')")
    print(format_engagement(report, args.metric, overall))

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metric": args.metric, "overall_mean": overall,
                       "synthetic": bool(args.synthetic),
                       "matched": len(joined), "tags": report}, f, indent=2)
        print(f"\nWrote report to {args.output}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="content_analysis_agent",
        description="Annotate product images with marketing tags.")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="structured (JSON lines) log verbosity")
    p.add_argument("--log-file", default=None,
                   help="write JSON log lines here instead of stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("taxonomy", help="print the controlled tag vocabulary")
    pt.set_defaults(func=_cmd_taxonomy)

    pg = sub.add_parser("tag", help="tag a folder of images")
    pg.add_argument("--input", required=True, help="folder of images (recursive)")
    pg.add_argument("--provider", default="anthropic",
                    choices=PROVIDERS)
    pg.add_argument("--model", default=None, help="override model id")
    pg.add_argument("--limit", type=int, default=None, help="max images")
    pg.add_argument("--output", default=None, help="results .json or .csv")
    pg.add_argument("--workers", type=int, default=1, metavar="N",
                    help="tag N images in parallel (network-bound work)")
    pg.add_argument("--few-shot", type=int, default=0, metavar="N",
                    help="prepend N labelled training images as examples")
    pg.add_argument("--train-dir", default="data/train",
                    help="where --few-shot examples come from")
    pg.add_argument("--memory", default=MEMORY_PATH,
                    help="path to the agent's tag memory (SQLite)")
    pg.add_argument("--no-memory", action="store_true",
                    help="ignore stored tags and always call the model")
    pg.add_argument("--enrich", action="store_true",
                    help="look the product up to justify non-visual tags "
                         "(awards, benchmark, energy rating)")
    pg.add_argument("--search-tool", default="mock",
                    choices=["mock", "anthropic"],
                    help="search backend for --enrich")
    pg.set_defaults(func=_cmd_tag)

    pe = sub.add_parser("eval", help="score the agent against train labels")
    pe.add_argument("--train-dir", required=True, help="labelled train folder")
    pe.add_argument("--provider", default="anthropic",
                    choices=PROVIDERS)
    pe.add_argument("--model", default=None, help="override model id")
    pe.add_argument("--sample", type=int, default=None,
                    help="evaluate only the first N labelled images")
    pe.add_argument("--report", default=None, help="write full report JSON")
    pe.add_argument("--no-baseline", action="store_true",
                    help="skip the agent-vs-baseline comparison table")
    pe.add_argument("--few-shot", type=int, default=0, metavar="N",
                    help="prepend up to N labelled examples, excluding the "
                         "image being scored")
    pe.add_argument("--memory", default=MEMORY_PATH,
                    help="path to the agent's tag memory (SQLite)")
    pe.add_argument("--no-memory", action="store_true",
                    help="ignore stored tags and always call the model")
    pe.add_argument("--enrich", action="store_true",
                    help="look the product up to justify non-visual tags "
                         "(awards, benchmark, energy rating)")
    pe.add_argument("--search-tool", default="mock",
                    choices=["mock", "anthropic"],
                    help="search backend for --enrich")
    pe.set_defaults(func=_cmd_eval)

    pi = sub.add_parser("insights",
                        help="join tags to the metadata sheet and rank tags "
                             "by engagement")
    src = pi.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="folder of images to tag, then analyse")
    src.add_argument("--results", help="reuse a results.json from `tag`")
    src.add_argument("--from-sheet", action="store_true",
                     help="take tags from the sheet's own labelled file names "
                          "(no model call; the supplied sheet covers the "
                          "labelled training images)")
    pi.add_argument("--metadata", default="data/meta_data.xlsx",
                    help="metadata sheet (.xlsx or .csv)")
    pi.add_argument("--synthetic", metavar="PATH", default=None,
                    help="generate a SYNTHETIC sheet at PATH and use it "
                         "(for demos when the real sheet is unavailable)")
    pi.add_argument("--metric", default="views",
                    help="numeric column to rank by (default: views)")
    pi.add_argument("--min-support", type=int, default=2,
                    help="ignore tags on fewer than N images")
    pi.add_argument("--provider", default="anthropic",
                    choices=PROVIDERS)
    pi.add_argument("--model", default=None)
    pi.add_argument("--limit", type=int, default=None)
    pi.add_argument("--workers", type=int, default=1)
    pi.add_argument("--output", default=None, help="write report JSON")
    pi.add_argument("--memory", default=MEMORY_PATH)
    pi.add_argument("--no-memory", action="store_true")
    pi.set_defaults(func=_cmd_insights)

    args = p.parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
