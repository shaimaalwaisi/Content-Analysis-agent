"""`tag` -- annotate a folder of images and write the results."""
from __future__ import annotations

import csv
import json

from agent.pipeline import results_to_dicts, run_folder
from agent.vlm import get_client

from .common import (add_enrich_args, add_memory_args, add_provider_args,
                     add_results_db_args, build_memory, build_search_tool,
                     build_store)
from .runlog import write_run


def _write(records: list[dict], path: str) -> None:
    """JSON by default; CSV when the filename asks for it."""
    if path.endswith(".csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "category", "model", "highlights",
                             "tags"])
            for r in records:
                writer.writerow([r["path"], r["category"], r["model"],
                                 "; ".join(r.get("highlights", [])),
                                 "; ".join(r["tags"])])
    else:
        with open(path, "w") as f:
            json.dump(records, f, indent=2)


def _tag_counts(records: list[dict]) -> dict:
    """How often each tag was predicted -- the quickest read on a run."""
    counts: dict[str, int] = {}
    for record in records:
        for tag in record["tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def run(args) -> None:
    from evaluation.runstats import RunStats
    from tools import new_run_id

    client = get_client(args.provider, args.model)

    examples = None
    if args.few_shot:
        from agent.fewshot import load_examples
        examples = load_examples(args.train_dir, limit=args.few_shot)
        print(f"Using {len(examples)} few-shot example(s) from {args.train_dir}")

    def progress(i, total, res):
        print(f"[{i}/{total}] {res.path.rsplit('/', 1)[-1]} -> {res.tags}")

    memory = build_memory(args)
    store = build_store(args)
    run_id = new_run_id() if store else None
    # The model id is what prices the run, so RunStats is told which one
    # answered rather than left to guess from the provider name.
    stats = RunStats(model_id=getattr(client, "model", ""))
    results = run_folder(args.input, client, limit=args.limit,
                         on_item=progress, examples=examples, memory=memory,
                         workers=args.workers,
                         search_tool=build_search_tool(args), stats=stats,
                         store=store, run_id=run_id)
    print(f"\n{stats.summary()}")
    if memory:
        print(memory.summary())
    if store:
        print(f"{store.summary()} as run {run_id} "
              f"-- view them in the Results tab of the Streamlit app")
        store.close()

    consistency = _consistency(args, client, results) if args.consistency \
        else None

    records = results_to_dicts(results)

    payload = {
        "images": len(records),
        "workflow": stats.as_dict(),
        "tag_counts": _tag_counts(records),
        "results": records,
    }
    if consistency:
        payload["consistency"] = consistency.as_dict()
    run_path = write_run("tag", args, payload)
    if run_path:
        print(f"Run record: {run_path}")

    if args.output:
        _write(records, args.output)
        print(f"\nWrote {len(records)} results to {args.output}")
    else:
        print(json.dumps(records, indent=2))


def _consistency(args, client, first):
    """Tag the same folder a second time and score the two passes against
    each other.

    Two things make the second pass an independent opinion rather than an echo
    of the first: a different draw of few-shot examples, and no memory. Without
    the second, an identical request would be served from the tag cache and
    every image would score a perfect 1.000 -- a number that measures the
    cache, not the model. The second pass also writes nothing to the results
    database: it exists to be compared, not to be shown to a content creator.
    """
    from agent.fewshot import load_examples
    from evaluation.consistency import compare_passes

    # Rotating the examples is what varies the question. With --few-shot 0
    # there are no examples to vary, so the only difference between the passes
    # is the model's own nondeterminism, which is a weaker test -- say so.
    examples = (load_examples(args.train_dir, limit=args.few_shot, seed=17)
                if args.few_shot else None)
    print(f"\nSecond pass for self-consistency ({len(first)} more model "
          f"call(s), which doubles the cost of this run)...")
    if not args.few_shot:
        print("  Note: with --few-shot 0 both passes ask exactly the same "
              "question, so this measures only the model's nondeterminism.")

    second = run_folder(args.input, client, limit=args.limit,
                        examples=examples, memory=None,
                        workers=args.workers,
                        search_tool=build_search_tool(args))
    report = compare_passes(first, second)
    print(f"\n{report.summary()}")
    return report


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
    p.add_argument("--consistency", action="store_true",
                   help="tag everything a second time with different examples "
                        "and score the agreement. Doubles the cost; needs no "
                        "labels, so it works on the test set.")
    add_provider_args(p)
    add_memory_args(p)
    add_enrich_args(p)
    add_results_db_args(p)
    p.set_defaults(func=run)
