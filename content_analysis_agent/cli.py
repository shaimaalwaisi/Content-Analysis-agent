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

from .pipeline import results_to_dicts, run_folder
from .taxonomy import allowed_tags, taxonomy_prompt
from .vlm import get_client


def _cmd_taxonomy(_args) -> None:
    tags = sorted(allowed_tags())
    print(taxonomy_prompt())
    print(f"\n{len(tags)} allowed tags total.")


def _cmd_tag(args) -> None:
    client = get_client(args.provider, args.model)

    def progress(i, total, res):
        name = res.path.rsplit("/", 1)[-1]
        print(f"[{i}/{total}] {name} -> {res.tags}")

    results = run_folder(args.input, client, limit=args.limit, on_item=progress)
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
    from .evaluate import evaluate  # local import: mock path needs no key

    client = get_client(args.provider, args.model)

    def progress(i, total, path, gt, got):
        name = path.rsplit("/", 1)[-1]
        mark = "OK " if set(gt) == set(got) else "xx "
        print(f"[{i}/{total}] {mark}{name}\n      truth={gt}\n      pred ={got}")

    metrics, records = evaluate(args.train_dir, client, sample=args.sample,
                                on_item=progress)
    print("\n" + "=" * 48)
    print(metrics.summary())
    print("=" * 48)

    if args.report:
        payload = {"metrics": {k: v for k, v in vars(metrics).items()
                               if k != "per_tag"},
                   "per_tag": metrics.per_tag, "records": records}
        with open(args.report, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull report written to {args.report}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="content_analysis_agent",
        description="Annotate product images with marketing tags.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("taxonomy", help="print the controlled tag vocabulary")
    pt.set_defaults(func=_cmd_taxonomy)

    pg = sub.add_parser("tag", help="tag a folder of images")
    pg.add_argument("--input", required=True, help="folder of images (recursive)")
    pg.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "mock"])
    pg.add_argument("--model", default=None, help="override model id")
    pg.add_argument("--limit", type=int, default=None, help="max images")
    pg.add_argument("--output", default=None, help="results .json or .csv")
    pg.set_defaults(func=_cmd_tag)

    pe = sub.add_parser("eval", help="score the agent against train labels")
    pe.add_argument("--train-dir", required=True, help="labelled train folder")
    pe.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "mock"])
    pe.add_argument("--model", default=None, help="override model id")
    pe.add_argument("--sample", type=int, default=None,
                    help="evaluate only the first N labelled images")
    pe.add_argument("--report", default=None, help="write full report JSON")
    pe.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
