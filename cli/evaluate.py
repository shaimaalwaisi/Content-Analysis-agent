"""`eval` -- score the agent against the labels in the training filenames."""
from __future__ import annotations

import json

from agent.vlm import get_client

from .common import (add_enrich_args, add_memory_args, add_provider_args,
                     build_memory, build_search_tool)
from .runlog import write_run


def run(args) -> None:
    # Imported here rather than at module scope so the mock path needs no key
    # and `taxonomy` does not pay for langgraph.
    from evaluation import (compare_baselines, evaluate, failure_warning,
                            format_comparison)
    from evaluation.runstats import RunStats

    client = get_client(args.provider, args.model)

    def progress(i, total, path, truth, got):
        mark = "OK " if set(truth) == set(got) else "xx "
        print(f"[{i}/{total}] {mark}{path.rsplit('/', 1)[-1]}"
              f"\n      truth={truth}\n      pred ={got}")

    if args.few_shot:
        print(f"Using up to {args.few_shot} few-shot example(s) "
              f"(the scored image is always excluded from its own examples)")

    memory = build_memory(args)
    stats = RunStats()
    metrics, records = evaluate(args.train_dir, client, sample=args.sample,
                                on_item=progress, few_shot=args.few_shot,
                                memory=memory,
                                search_tool=build_search_tool(args),
                                stats=stats)

    # A failed run must not read as a bad score, so this comes first.
    warning = failure_warning(records)
    if warning:
        print(warning)
    print("\n" + "=" * 48)
    print(metrics.summary())
    if memory:
        print(memory.summary())
    print("=" * 48)
    print("\n" + stats.summary())

    scored: dict = {}
    if not args.no_baseline:
        scored = compare_baselines([r["truth"] for r in records],
                                   [r["predicted"] for r in records])
        print("\nAgent vs model-free baselines")
        print(format_comparison(scored))
        if warning:
            print("\n(Scores above are unreliable - see the failure warning.)")

    payload = {
            "metrics": {k: v for k, v in vars(metrics).items()
                        if k != "per_tag"},
            "workflow": stats.as_dict(),
            "per_tag": metrics.per_tag,
            "baselines": {name: {k: v for k, v in vars(m).items()
                                 if k != "per_tag"}
                          for name, m in scored.items() if name != "agent"},
        "records": records,
    }

    run_path = write_run("eval", args, payload)
    if run_path:
        print(f"\nRun record: {run_path}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Full report also written to {args.report}")


def add_parser(sub) -> None:
    p = sub.add_parser("eval", help="score the agent against train labels")
    p.add_argument("--train-dir", required=True, help="labelled train folder")
    p.add_argument("--sample", type=int, default=None,
                   help="evaluate only the first N labelled images")
    p.add_argument("--report", default=None, help="write full report JSON")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the agent-vs-baseline comparison table")
    p.add_argument("--few-shot", type=int, default=0, metavar="N",
                   help="prepend up to N labelled examples, excluding the "
                        "image being scored")
    add_provider_args(p)
    add_memory_args(p)
    add_enrich_args(p)
    p.set_defaults(func=run)
