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
from agent.vlm import DEFAULT_MODEL, get_client  # noqa: E402

setup_logging(os.environ.get("LOG_LEVEL", "WARNING"))

# Wide, because the results table is seven columns of text: the default
# centred layout gives it about half the window and wraps every sentence.
st.set_page_config(page_title="Content Analysis Agent", page_icon="🏷️",
                   layout="wide")
st.title("🏷️ Content Analysis Agent")
st.caption("Annotate retail product images with marketing tags. Sony AI task MVP.")

with st.sidebar:
    st.header("Settings")
    # One provider, so there is nothing to choose: stating it beats a
    # dropdown with a single entry.
    st.caption(f"Model: `{DEFAULT_MODEL}`")
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

MAX_UPLOAD = 10          # what the results table is sized for

# Columns worth showing when there is something in them, and worth dropping
# when there is not. Image name, Highlights and Marketing tags always stay:
# they are what the table is for, and an empty Highlights cell is a fact about
# the image rather than a gap in what we know.
ALWAYS_OPTIONAL = ("Product", "Category", "Description", "Specs",
                   "Price", "Views")

# How much of the width each column deserves. Sentences need room; a category
# or a price does not, and left to itself the table gives them equal shares.
COLUMN_WIDTHS = {
    "Image name": "medium", "Product": "small", "Category": "small",
    "Description": "large", "Specs": "large", "Highlights": "medium",
    "Marketing tags": "large", "Price": "small", "Views": "small",
    "How sure? (confidence)": "small", "Why (signals)": "medium",
}


def _confidence(row: dict) -> tuple[str, str]:
    """How much to trust one image's tags, and why -- from what we stored.

    There is no ground truth for an uploaded image, so this is confidence, not
    correctness, and it is labelled that way on screen. It reads the three
    signals the run already recorded:

      * the reasoning loop asked twice (`attempts > 1`), meaning the first
        answer was weak enough to re-prompt;
      * the model produced no reason for a tag it kept, so nothing supports it;
      * only one tag came back, which on this taxonomy usually means the model
        found nothing beyond the obvious.

    None of these proves a tag is wrong. They are what an experienced reviewer
    would look at first, which is the whole job of this column: decide which
    rows are worth a human's eyes.
    """
    tags = row.get("tags") or []
    reasons = row.get("rationale") or {}
    warnings = []
    if (row.get("attempts") or 1) > 1:
        warnings.append("re-prompted")
    if tags and not all(t in reasons for t in tags):
        # A cached row from a plain client has no reasons at all; that is a
        # property of the client, not a doubt about the image.
        if reasons:
            warnings.append("some tags unexplained")
    if len(tags) == 1:
        warnings.append("only one tag")
    if not tags:
        return "Needs a look", "nothing was tagged"
    if not warnings:
        return "High", "clean first answer, every tag explained"
    if len(warnings) == 1:
        return "Medium", warnings[0]
    return "Needs a look", ", ".join(warnings)


# The latency table is read by people who did not write the agent, so every
# heading and row label leads with plain English and keeps the technical term
# in brackets -- the bracketed word is what the CLI and the run record use, so
# the two are still recognisably the same number.
ACTION_LABELS = {
    "encode": "Prepare the image (encode)",
    "model": "Ask the model (model call)",
    "search": "Look the product up (search)",
}

# A description is a sentence and specs are a list: at the default 35px they
# are truncated to a few words. Taller rows let the grid wrap them.
ROW_PIXELS = 64
HEADER_PIXELS = 42


def _session_run() -> str:
    """The run id for this browser session, created on first use.

    Scoping uploads to a session run is what keeps the Results tab honest:
    it shows the images *you* tagged, not whatever a previous CLI run left in
    the database.
    """
    from tools import new_run_id
    if "run_id" not in st.session_state:
        st.session_state.run_id = f"ui-{new_run_id()}"
    return st.session_state.run_id


def _session_stats(model_id: str):
    """The workflow metrics for this browser session's tagging.

    Scoped to the session for the same reason the Results tab is: the numbers
    must describe what *you* just tagged, not whatever a CLI run left behind.
    Switching model mid-session starts them over -- one cost per task cannot
    describe two price cards.
    """
    from evaluation import RunStats
    stats = st.session_state.get("stats")
    if stats is None or (stats.tasks and stats.model_id != model_id):
        stats = RunStats(model_id=model_id)
        st.session_state.stats = stats
    stats.model_id = model_id
    return stats


def _context_for(name: str) -> str | None:
    """Product context for an uploaded file, if anything knows any.

    A CLI run reads category and product off the folder path. An upload has no
    path, and the sidebar no longer asks, so the metadata sheet is the only
    source left -- and it only has a row for the labelled training images.
    Everything else is tagged with no context at all, which the agent handles:
    context sharpens the prompt, it is not required by it.
    """
    meta = _metadata_rows().get(name)
    if not meta:
        return None
    bits = []
    if meta.get("category"):
        bits.append(f"Category: {meta['category']}")
    if meta.get("product"):
        bits.append(f"Model: {meta['product']}")
    return ", ".join(bits) or None


def _tagging_tab() -> None:
    """Upload up to ten product images and tag them with the live agent."""
    from tools import RESULTS_PATH, ResultStore

    upload_col, _spacer = st.columns([3, 2])
    uploads = upload_col.file_uploader(
        f"Upload up to {MAX_UPLOAD} product images",
        type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
        help="Each image goes through the same agent the CLI runs.")

    if not uploads:
        st.caption("Tagging calls Claude, so ANTHROPIC_API_KEY must be set "
                   "(see .env.example).")
        return

    if len(uploads) > MAX_UPLOAD:
        st.warning(f"{len(uploads)} images selected; tagging the first "
                   f"{MAX_UPLOAD}.")
        uploads = uploads[:MAX_UPLOAD]

    # A contact sheet of what is about to be tagged. Fixed-width columns, so
    # a batch of three does not stretch three thumbnails across the window.
    per_row = 5
    for start in range(0, len(uploads), per_row):
        columns = st.columns(per_row)
        for col, item in zip(columns, uploads[start:start + per_row]):
            col.image(item, caption=item.name, width="stretch")

    button_col, _spacer = st.columns([1, 5])
    if not button_col.button(f"Tag {len(uploads)} image(s)", type="primary",
                             width="stretch"):
        return

    examples = None
    if few_shot:
        from agent.fewshot import load_examples
        examples = load_examples(limit=few_shot)

    run_id = _session_run()
    store = ResultStore(RESULTS_PATH)
    progress = st.progress(0.0, text="Tagging...")
    failures = []
    try:
        client = get_client()
        stats = _session_stats(getattr(client, "model", ""))
        memory = TagMemory() if use_memory else None
        app = build_graph(client, memory=memory, store=store, run_id=run_id,
                          stats=stats)
        # Keep each upload's own name on disk: it is what the results table
        # shows, and what the metadata sheet is joined on.
        workspace = tempfile.mkdtemp(prefix="upload-")
        for i, item in enumerate(uploads, 1):
            path = os.path.join(workspace, os.path.basename(item.name))
            with open(path, "wb") as handle:
                handle.write(item.getvalue())
            stats.record_task()
            tags = []
            try:
                out = app.invoke({"image_path": path,
                                  "context": _context_for(item.name),
                                  "examples": examples, "run_id": run_id})
                tags = out.get("tags", [])
            except Exception as exc:       # one bad image must not stop the
                failures.append((item.name, str(exc)))   # rest of the batch
                stats.record_failure()
            finally:
                # Outside the except: an image that raised and an image that
                # came back empty are both tasks that returned no tags.
                stats.record_outcome(len(tags))
                os.unlink(path)
            progress.progress(i / len(uploads), text=f"Tagged {i} of "
                                                     f"{len(uploads)}")
        os.rmdir(workspace)
    except Exception as exc:
        st.error(f"Tagging failed: {exc}")
        return
    finally:
        progress.empty()
        store.close()

    for name, error in failures:
        st.error(f"{name}: {error}")
    tagged = len(uploads) - len(failures)
    if tagged:
        st.success(f"Tagged {tagged} image(s). See the Results and "
                   f"Metrics tabs.")
        _render_results(run_id, key="tagging")


def _results_tab() -> None:
    """The content creator's view: the images this session tagged, as a table.

    Reads the results database only -- no model calls. Price and view counts
    come from the metadata sheet at render time, so fixing the sheet never
    means re-tagging.
    """
    from tools import RESULTS_PATH, ResultStore

    run_id = st.session_state.get("run_id")
    if run_id is None:
        st.info("Nothing tagged yet. Upload images in the **Tag images** tab "
                "and press Tag; they appear here.")

    if not os.path.exists(RESULTS_PATH):
        return
    if st.checkbox("Show earlier runs", value=False,
                   help="Runs from other sessions and from `python -m cli "
                        "tag`, newest first"):
        store = ResultStore(RESULTS_PATH)
        try:
            runs = store.runs()
        finally:
            store.close()
        if not runs:
            st.caption("The results database is empty.")
            return
        labels = {f"{r['run_id']} ({r['images']} images)": r["run_id"]
                  for r in runs}
        run_id = labels[st.selectbox("Run", list(labels), index=0)]
    if run_id:
        _render_results(run_id)


def _render_results(run_id: str, key: str = "results") -> None:
    """One run as a table, with the model's reasons underneath.

    Both tabs can render the same run in one pass -- the Tag tab shows the
    batch it just finished, the Results tab shows the session -- so the
    download button is namespaced by caller to keep the widget ids unique.
    """
    from tools import RESULTS_PATH, ResultStore

    store = ResultStore(RESULTS_PATH)
    try:
        rows = store.rows(run_id)
    finally:
        store.close()
    if not rows:
        st.caption("No images in this run.")
        return

    meta = _metadata_rows()
    by_product = _product_rows()
    table, from_product = [], 0
    for row in rows:
        # The sheet is keyed on file name; failing that, on the product, whose
        # other images still say what it costs and how much it is looked at.
        extra = meta.get(row["image_name"], {})
        verdict, why = _confidence(row)
        product = row["product"] or extra.get("product", "")
        if not extra:
            extra = by_product.get(str(product).strip().upper(), {})
            if extra:
                from_product += 1
        table.append({
            "Image name": row["image_name"],
            "Product": product,
            "Category": row["category"] or extra.get("category", ""),
            "Description": row.get("description", ""),
            "Specs": row.get("specs", ""),
            "Highlights": ", ".join(row["highlights"]),
            "Marketing tags": ", ".join(row["tags"]),
            "How sure? (confidence)": verdict,
            "Why (signals)": why,
            "Price": _number(extra.get("price")),
            "Views": _number(extra.get("views")),
        })

    # A column of nothing but blanks reads as a bug rather than as an absence:
    # price and views exist only in meta_data.xlsx, which has no row for most
    # images, and the model leaves Product empty rather than guessing a name
    # the image does not print. So a column that answered for no image in the
    # batch is dropped, and named underneath instead.
    hidden = [column for column in ALWAYS_OPTIONAL
              if not any(record[column] for record in table)]
    for record in table:
        for column in hidden:
            del record[column]

    needs_a_look = sum(1 for r in table
                       if r["How sure? (confidence)"] == "Needs a look")
    c1, c2, c3, c4, _spacer = st.columns([1, 1, 1, 1, 4])
    c1.metric("Images", len(rows))
    c2.metric("Worth checking", needs_a_look,
              help="Images whose tags showed more than one warning sign. "
                   "Start your review here — the rest are likely fine.")
    c3.metric("Reused, no AI call (from memory)",
              sum(1 for r in rows if r["cached"]),
              help="Rows served from the agent's cache. The same image, "
                   "already tagged under the same settings, costs nothing "
                   "the second time.")
    c4.metric("Asked twice (re-prompted)",
              sum(1 for r in rows if (r["attempts"] or 0) > 1),
              help="Images where the first answer looked weak — nothing "
                   "survived the taxonomy, or more tags were rejected than "
                   "kept — so the agent asked the model again.")

    # Sized to the batch: ten rows fit without a scrollbar inside the table,
    # which is the whole point of showing them as a table.
    st.dataframe(
        table, width="stretch", hide_index=True, row_height=ROW_PIXELS,
        height=ROW_PIXELS * len(table) + HEADER_PIXELS + 3,
        column_config={name: st.column_config.TextColumn(
            name, width=COLUMN_WIDTHS.get(name, "medium"),
            pinned=(name == "Image name"))
            for name in table[0]})

    if from_product:
        st.caption(f"Price and Views for {from_product} image(s) are the "
                   f"median across that product's rows in meta_data.xlsx, "
                   f"not that image's own figures.")
    if hidden:
        from_sheet = [c for c in hidden if c in ("Price", "Views")]
        from_image = [c for c in hidden if c not in ("Price", "Views")]
        why = []
        if from_sheet:
            why.append(f"{' and '.join(from_sheet)} come from "
                       f"meta_data.xlsx, which has no row for these images")
        if from_image:
            why.append(f"{', '.join(from_image)} — nothing the model could "
                       f"read off them, and nothing set in the sidebar")
        st.caption(f"Hidden, empty for every image in this run: "
                   f"{', '.join(hidden)}. " + "; ".join(why) + ".")

    untagged = [r["image_name"] for r in rows if not r["tags"]]
    if untagged:
        st.warning(f"{len(untagged)} image(s) came back with no tags: "
                   f"{', '.join(untagged[:5])}")

    with st.expander("Why these tags? (the model's own reasons)"):
        explained = [r for r in rows if r["rationale"]]
        if not explained:
            st.caption("No reasons recorded. The mock client does not "
                       "inspect the image, so it has nothing to explain.")
        for row in explained:
            st.markdown(f"**{row['image_name']}**")
            for tag, why in row["rationale"].items():
                st.caption(f"`{tag}` - {why}")

    st.download_button("Download as CSV", _to_csv(table),
                       file_name=f"{run_id}_tags.csv", mime="text/csv",
                       key=f"csv-{key}-{run_id}")


@st.cache_data(show_spinner=False)
def _product_rows() -> dict:
    """model name -> what the sheet knows about that product.

    The sheet has no row for most uploads: it documents the training images,
    whose file names are tag lists. But it does carry many rows per product,
    so when the file name is unknown and the model is not, the product's own
    median price and median views are still a real answer -- labelled as such
    under the table, never presented as this image's own view count.
    """
    from statistics import median

    def numbers(values):
        out = []
        for v in values:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:               # NaN is the only value unequal to itself
                out.append(f)
        return out

    grouped: dict[str, list] = {}
    for rec in _metadata_rows().values():
        key = str(rec.get("product") or "").strip().upper()
        if key:
            grouped.setdefault(key, []).append(rec)
    out = {}
    for key, recs in grouped.items():
        prices = numbers(r.get("price") for r in recs)
        views = numbers(r.get("views") for r in recs)
        out[key] = {"category": recs[0].get("category", ""),
                    "price": median(prices) if prices else "",
                    "views": median(views) if views else "",
                    "images": len(recs)}
    return out


@st.cache_data(show_spinner=False)
def _metadata_rows() -> dict:
    """file name -> its metadata sheet row, or {} when the sheet is absent.

    The sheet ships with the task rather than the repo, and both the table and
    the per-upload context are still useful without it, so a missing sheet
    costs two columns and nothing else. Cached: it is read once per upload and
    once per render, and it does not change while the app is running.
    """
    path = os.environ.get("METADATA_PATH", "data/meta_data.xlsx")
    if not os.path.exists(path):
        return {}
    try:
        from data.metadata import load_metadata
        return {rec["file"]: rec for rec in load_metadata(path)}
    except Exception:             # pandas missing, unreadable sheet, ...
        return {}


def _number(value) -> str:
    """A sheet number as display text, or "" when there is nothing to show.

    Kept as text on purpose: a column holding both floats and blanks cannot be
    serialised for the table, and every row here is either a number or a gap.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number:             # NaN
        return ""
    return f"{number:,.0f}"


def _to_csv(table: list[dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    if table:
        writer = csv.DictWriter(buf, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return buf.getvalue()


def _metrics_tab() -> None:
    """The three workflow metrics for what this session tagged.

    Session-scoped on purpose. Reading the newest run record instead would put
    a previous CLI run's numbers on screen before the user has tagged
    anything, which reads as a measurement of a run that never happened.
    """
    stats = st.session_state.get("stats")
    if stats is None or not stats.tasks:
        st.info("Nothing tagged yet. Upload images in the **Tag images** tab "
                "and press Tag; the metrics for that run appear here.")
        st.caption("Task success rate, cost per task, and latency per action "
                   "— measured on your own run, with no labels needed.")
    else:
        _render_metrics(stats.as_dict(),
                        caption=f"This session: {stats.tasks} image(s) tagged "
                                f"with {stats.model_id or 'the mock client'}.")

    # A CLI run is a different run, so it is opt-in and clearly labelled --
    # never the thing shown by default.
    from cli.runlog import latest
    newest = latest("tag")
    if not newest:
        return
    st.divider()
    if st.checkbox(f"Show a `python -m cli tag` run instead",
                   value=False, help="Run records written into results/"):
        path = st.text_input("Run record", value=newest)
        if not os.path.exists(path):
            st.warning(f"No run record at `{path}`.")
            return
        try:
            with open(path) as handle:
                report = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            st.error(f"Could not read {path}: {exc}")
            return
        workflow = report.get("workflow", {})
        if not workflow:
            st.warning(f"`{os.path.basename(path)}` carries no workflow "
                       f"metrics.")
            return
        _render_metrics(workflow,
                        caption=f"From `{os.path.basename(path)}` — a "
                                f"command-line run, not this session.")
        _render_consistency(report.get("consistency"))


def _render_consistency(report: dict | None) -> None:
    """Self-consistency, when the run was made with `--consistency`.

    Kept visually apart from the three workflow metrics: it answers a
    different question (is the model steady?) and costs a second model call
    per image, so a run without it is the normal case, not a broken one.
    """
    if not report:
        st.caption("This run has no self-consistency check. Add "
                   "`--consistency` to `python -m cli tag` to tag everything "
                   "twice and score the agreement — it doubles the cost.")
        return
    st.caption("Does the model say the same thing twice? (self-consistency)")
    c1, c2, _spacer = st.columns([1, 1, 6])
    c1.metric("Answers that matched (mean agreement)",
              f"{report.get('mean_agreement', 0.0):.3f}",
              help="Each image was tagged twice with different examples. "
                   "1.000 means both passes produced exactly the same tags. "
                   "This is steadiness, not correctness — a model can be "
                   "repeatably wrong.")
    c2.metric("Changed its mind (unstable)", report.get("unstable", 0),
              help=f"Images the two passes agreed on less than "
                   f"{report.get('threshold', 0.7):.0%} of the tags for. "
                   f"These are the ones worth a human's eyes.")

    unstable = sorted((r for r in report.get("per_image", [])
                       if r.get("agreement", 1.0) < report.get("threshold",
                                                               0.7)),
                      key=lambda r: r.get("agreement", 0.0))
    if not unstable:
        return
    st.dataframe(
        [{"Image": r["image"],
          "Agreed on (score)": r["agreement"],
          "First answer": ", ".join(r["first"]) or "—",
          "Second answer": ", ".join(r["second"]) or "—"}
         for r in unstable],
        width="stretch", hide_index=True,
        column_config={
            "Image": st.column_config.TextColumn("Image", width="medium"),
            "Agreed on (score)": st.column_config.NumberColumn(
                "Agreed on (score)", width="small", format="%.2f",
                help="1.00 = identical tags both times, 0.00 = nothing in "
                     "common"),
            "First answer": st.column_config.TextColumn(
                "First answer", width="large"),
            "Second answer": st.column_config.TextColumn(
                "Second answer", width="large"),
        })


def _render_metrics(workflow: dict, caption: str) -> None:
    """The three metrics, from either this session or a saved run record."""
    st.caption(caption)
    cost = workflow.get("cost_per_task_usd")
    c1, c2, _spacer = st.columns([1, 1, 6])
    c1.metric("Task success", f"{workflow.get('task_success_rate', 0.0):.3f}",
              help=f"{workflow.get('successes', 0)} of "
                   f"{workflow.get('tasks', 0)} images came back with at "
                   f"least one tag ({workflow.get('failures', 0)} error(s))")
    # An unpriced model shows a dash, not a zero: having no rate card is not
    # the same fact as being free, and 0.00 would be read as free.
    c2.metric("Cost per task", "—" if cost is None else f"${cost:.5f}",
              help=f"{workflow.get('input_tokens', 0):,} in / "
                   f"{workflow.get('output_tokens', 0):,} out tokens on "
                   f"{workflow.get('model_id') or 'an unknown model'}; "
                   f"{workflow.get('cache_hits', 0)} task(s) served from "
                   f"cache at no cost")

    st.caption("How long each step took (latency per action)")
    latency = workflow.get("latency_per_action_ms", {})
    if not latency:
        st.caption("No steps were timed in this run.")
        return
    st.dataframe(
        [{"Step (action)": ACTION_LABELS.get(name, name),
          "Times run (calls)": at.get("calls", 0),
          "Usual time (p50 ms)": at.get("p50", 0),
          "Slow ones (p95 ms)": at.get("p95", 0)}
         for name, at in latency.items()],
        width="stretch", hide_index=True,
        column_config={
            "Step (action)": st.column_config.TextColumn(
                "Step (action)", width="medium",
                help="One thing the agent does per image"),
            "Times run (calls)": st.column_config.NumberColumn(
                "Times run (calls)", width="small",
                help="How many times this step ran across the whole batch"),
            "Usual time (p50 ms)": st.column_config.NumberColumn(
                "Usual time (p50 ms)", width="small",
                help="The middle value in milliseconds: half the runs were "
                     "faster than this, half slower (the median, p50)"),
            "Slow ones (p95 ms)": st.column_config.NumberColumn(
                "Slow ones (p95 ms)", width="small",
                help="In milliseconds, 19 runs out of 20 finished faster than "
                     "this. It is the wait a user actually notices, which is "
                     "why it is shown next to the usual time (the 95th "
                     "percentile, p95)"),
        })


tag_tab, results_tab, metrics_tab = st.tabs(
    ["Tag images", "Results", "Metrics"])
with tag_tab:
    _tagging_tab()
with results_tab:
    _results_tab()
with metrics_tab:
    _metrics_tab()
