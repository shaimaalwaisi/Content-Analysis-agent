"""Simple Streamlit demo for the content tagging agent.

Runs the SAME LangGraph agent as the CLI. Start it from the repo root:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Make the package importable when launched via `streamlit run`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from agent.graph import build_graph  # noqa: E402
from agent.logconf import setup_logging  # noqa: E402
from agent.memory import TagMemory  # noqa: E402
from agent.taxonomy import taxonomy_prompt  # noqa: E402
from agent.vlm import PROVIDERS, get_client  # noqa: E402

setup_logging(os.environ.get("LOG_LEVEL", "WARNING"))

st.set_page_config(page_title="Content Analysis Agent", page_icon="🏷️")
st.title("🏷️ Content Analysis Agent")
st.caption("Annotate retail product images with marketing tags. Sony AI task MVP.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", PROVIDERS, index=0)
    model = st.text_input("Model (optional)", value="")
    context = st.text_input("Product context (optional)",
                            placeholder="Category: Mobile, Model: XPERIA10MK5")
    few_shot = st.slider(
        "Few-shot examples", 0, 8, 0,
        help="Show the model this many labelled training images before "
             "asking. 0 = zero-shot.")
    use_memory = st.checkbox(
        "Reuse remembered tags", value=True,
        help="Skip the model when this exact image has been tagged before "
             "under the same settings.")
    with st.expander("Controlled vocabulary"):
        st.text(taxonomy_prompt())

def _tagging_tab() -> None:
    """Upload one image and tag it with the live agent."""
    uploaded = st.file_uploader("Upload a product image",
                                type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, width="stretch")
        if st.button("Tag image", type="primary"):
            # Write the upload to a temp file so the on-disk agent can run it.
            suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                client = get_client(provider, model or None)
                memory = TagMemory() if use_memory else None
                app = build_graph(client, memory=memory)
                examples = None
                if few_shot:
                    from agent.fewshot import load_examples
                    examples = load_examples(limit=few_shot)
                with st.spinner("Tagging..."):
                    out = app.invoke({"image_path": tmp_path,
                                      "context": context or None,
                                      "examples": examples})
                tags = out.get("tags", [])
                from_memory = out.get("cached", False)
            except Exception as exc:
                st.error(f"Tagging failed: {exc}")
                tags, from_memory = [], False
            finally:
                os.unlink(tmp_path)

            if tags:
                st.success("Predicted tags")
                st.write(" ".join(f"`{t}`" for t in tags))
                if from_memory:
                    st.caption("Reused from memory - no model call was made.")
            else:
                st.warning("No tags predicted.")
    st.divider()
    st.caption("Tip: set provider = mock to try the UI with no API key.")


def _results_tab() -> None:
    """The content creator's view: every image a run tagged, as one table.

    Reads the results database only -- no model calls. Price and view counts
    come from the metadata sheet at render time, so fixing the sheet never
    means re-tagging.
    """
    from tools import RESULTS_PATH, ResultStore

    db_path = st.text_input(
        "Results database", value=RESULTS_PATH,
        help="Written by `python -m cli tag`. One row per image per run.")
    if not os.path.exists(db_path):
        st.info(f"No results database at `{db_path}` yet. Tag a batch with:")
        st.code("python -m cli tag --input data/test --limit 10 "
                "--provider anthropic --few-shot 8", language="bash")
        return

    store = ResultStore(db_path)
    try:
        runs = store.runs()
        if not runs:
            st.info("The database is empty. Run `python -m cli tag` first.")
            return
        labels = {f"{r['run_id']} ({r['images']} images)": r["run_id"]
                  for r in runs}
        chosen = st.selectbox("Run", list(labels), index=0)
        rows = store.rows(labels[chosen])
    finally:
        store.close()

    meta = _metadata_rows()
    table = []
    for row in rows:
        extra = meta.get(row["image_name"], {})
        table.append({
            "Image name": row["image_name"],
            "Product": row["product"] or extra.get("product", ""),
            "Category": row["category"] or extra.get("category", ""),
            "Highlights": ", ".join(row["highlights"]),
            "Price": extra.get("price", ""),
            "Views": extra.get("views", ""),
            "Marketing tags": ", ".join(row["tags"]),
        })

    c1, c2, c3 = st.columns(3)
    c1.metric("Images", len(rows))
    c2.metric("From memory", sum(1 for r in rows if r["cached"]),
              help="Rows served from the agent's cache, with no model call")
    c3.metric("Re-prompted", sum(1 for r in rows if (r["attempts"] or 0) > 1),
              help="Images the reasoning loop looked at twice")

    st.dataframe(table, width="stretch", hide_index=True)

    untagged = [r["image_name"] for r in rows if not r["tags"]]
    if untagged:
        st.warning(f"{len(untagged)} image(s) came back with no tags: "
                   f"{', '.join(untagged[:5])}")

    with st.expander("Why these tags? (the model's own reasons)"):
        explained = [r for r in rows if r["rationale"]]
        if not explained:
            st.caption("No reasons recorded. The mock client and plain "
                       "clients do not produce them.")
        for row in explained:
            st.markdown(f"**{row['image_name']}**")
            for tag, why in row["rationale"].items():
                st.caption(f"`{tag}` - {why}")

    st.download_button("Download as CSV", _to_csv(table),
                       file_name=f"{labels[chosen]}_tags.csv",
                       mime="text/csv")


def _metadata_rows() -> dict:
    """file name -> its metadata sheet row, or {} when the sheet is absent.

    The sheet ships with the task rather than the repo, and the table is still
    useful without it, so a missing sheet costs two columns and nothing else.
    """
    path = os.environ.get("METADATA_PATH", "data/meta_data.xlsx")
    if not os.path.exists(path):
        return {}
    try:
        from analysis.metadata import load_metadata
        return {rec["file"]: rec for rec in load_metadata(path)}
    except Exception as exc:      # pandas missing, unreadable sheet, ...
        st.caption(f"Metadata sheet not joined: {exc}")
        return {}


def _to_csv(table: list[dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    if table:
        writer = csv.DictWriter(buf, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return buf.getvalue()


def _scores_tab() -> None:
    """Read-only view of a report written by `python -m cli eval --report`.

    Deliberately makes no model calls: evaluation is a minutes-long batch job
    that belongs on the command line, and re-running it from a web page would
    add a progress UI without adding information. This just renders the JSON
    the CLI already produces.
    """
    from cli.runlog import latest
    from evaluation import TARGET_EXACT_MATCH, TARGET_MICRO_F1

    # Every `cli eval` run drops a record in results/; show the newest.
    default = latest("eval") or "metrics.json"
    path = st.text_input("Report file", value=default,
                         help="Every `python -m cli eval` run writes one of "
                              "these into results/")
    if not os.path.exists(path):
        st.info(f"No report at `{path}` yet. Generate one with:")
        st.code("python -m cli eval --train-dir data/train "
                "--provider anthropic --few-shot 8", language="bash")
        return

    try:
        with open(path) as handle:
            report = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        st.error(f"Could not read {path}: {exc}")
        return

    metrics = report.get("metrics", {})
    failed = [r for r in report.get("records", []) if r.get("error")]
    if failed:
        st.error(f"{len(failed)} of {len(report.get('records', []))} images "
                 f"failed in this run, and scored as empty predictions. "
                 f"The numbers below understate the model.")
        with st.expander("Failures"):
            for r in failed[:10]:
                st.caption(f"{os.path.basename(r['path'])} - {r['error'][:200]}")

    st.subheader("Tagging quality")
    micro_f1 = metrics.get("micro_f1", 0.0)
    exact = metrics.get("exact_match", 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Micro-F1", f"{micro_f1:.3f}",
              f"{micro_f1 - TARGET_MICRO_F1:+.3f} vs target")
    c2.metric("Exact match", f"{exact:.3f}",
              f"{exact - TARGET_EXACT_MATCH:+.3f} vs target")
    c3.metric("Jaccard", f"{metrics.get('jaccard', 0.0):.3f}")
    c4.metric("Images", metrics.get("n", 0))

    baselines = report.get("baselines", {})
    if baselines:
        st.subheader("Against model-free baselines")
        rows = [{"run": "agent", "micro-F1": round(micro_f1, 3),
                 "Jaccard": round(metrics.get("jaccard", 0.0), 3),
                 "exact": round(exact, 3)}]
        rows += [{"run": name, "micro-F1": round(m.get("micro_f1", 0.0), 3),
                  "Jaccard": round(m.get("jaccard", 0.0), 3),
                  "exact": round(m.get("exact_match", 0.0), 3)}
                 for name, m in baselines.items()]
        st.dataframe(rows, width="stretch", hide_index=True)
        best = max((m.get("micro_f1", 0.0) for m in baselines.values()),
                   default=0.0)
        verdicts = [
            ("Beats best baseline", micro_f1 > best,
             f"{micro_f1:.3f} vs {best:.3f}"),
            (f"Micro-F1 >= {TARGET_MICRO_F1:.2f}", micro_f1 >= TARGET_MICRO_F1,
             f"{micro_f1:.3f}"),
            (f"Exact-match >= {TARGET_EXACT_MATCH:.2f}",
             exact >= TARGET_EXACT_MATCH, f"{exact:.3f}"),
        ]
        for label, passed, detail in verdicts:
            st.write(f"{'PASS' if passed else 'FAIL'} - {label} ({detail})")

    workflow = report.get("workflow", {})
    if workflow:
        st.subheader("Agent workflow (no labels required)")
        w1, w2, w3 = st.columns(3)
        w1.metric("Hallucination",
                  f"{workflow.get('hallucination_rate', 0.0):.3f}",
                  help="Share of proposed tags outside the vocabulary")
        w2.metric("Model p95", f"{workflow.get('model_ms_p95', 0):.0f} ms")
        w3.metric("Cache hit", f"{workflow.get('cache_hit_rate', 0.0):.3f}")

    per_tag = report.get("per_tag", {})
    if per_tag:
        st.subheader("Per tag")
        rows = [{"tag": tag, **stats} for tag, stats in
                sorted(per_tag.items(), key=lambda kv: -kv[1].get("support", 0))]
        st.dataframe(rows, width="stretch", hide_index=True)


tag_tab, results_tab, scores_tab = st.tabs(
    ["Tag an image", "Results", "Scores"])
with tag_tab:
    _tagging_tab()
with results_tab:
    _results_tab()
with scores_tab:
    _scores_tab()
