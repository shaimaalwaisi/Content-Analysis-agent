"""`tag` -- annotate a folder of images and write the results."""
from __future__ import annotations

import csv
import json

from content_analysis_agent.pipeline import results_to_dicts, run_folder
from content_analysis_agent.vlm import get_client

from .common import (add_enrich_args, add_memory_args, add_provider_args,
                     build_memory, build_search_tool)


def _write(records: list[dict], path: str) -> None:
    """JSON by default; CSV when the filename asks for it."""
    if path.endswith(".csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "category", "model", "tags"])
            for r in records:
                writer.writerow([r["path"], r["category"], r["model"],
                                 "; ".join(r["tags"])])
    else:
        with open(path, "w") as f:
            json.dump(records, f, indent=2)


def run(args) -> None:
    from evaluation.runstats import RunStats

    client = get_client(args.provider, args.model)

    examples = None
    if args.few_shot:
        from content_analysis_agent.fewshot import load_examples
        examples = load_examples(args.train_dir, limit=args.few_shot)
        print(f"Using {len(examples)} few-shot example(s) from {args.train_dir}")

    def progress(i, total, res):
        print(f"[{i}/{total}] {res.path.rsplit('/', 1)[-1]} -> {res.tags}")

    memory = build_memory(args)
    stats = RunStats()
    results = run_folder(args.input, client, limit=args.limit,
                         on_item=progress, examples=examples, memory=memory,
                         workers=args.workers,
                         search_tool=build_search_tool(args), stats=stats)
    print(f"\n{stats.summary()}")
    if memory:
        print(memory.summary())

    records = results_to_dicts(results)
    if args.output:
        _write(records, args.output)
        print(f"\nWrote {len(records)} results to {args.output}")
    else:
        print(json.dumps(records, indent=2))


def add_parser(sub) -> None:
    p = sub.add_parser("tag", help="tag a folder of images")
    p.add_argument("--input", required=True,
                   help="folder of images (searched recursively)")
    p.add_argument("--limit", type=int, default=None, help="max images")
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
    p.set_defaults(func=run)
