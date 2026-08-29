"""Simple Streamlit demo for the content tagging agent.

Runs the SAME LangGraph agent as the CLI. Start it from the repo root:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

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

from content_analysis_agent.graph import build_graph  # noqa: E402
from content_analysis_agent.logconf import setup_logging  # noqa: E402
from content_analysis_agent.memory import TagMemory  # noqa: E402
from content_analysis_agent.taxonomy import taxonomy_prompt  # noqa: E402
from content_analysis_agent.vlm import PROVIDERS, get_client  # noqa: E402

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
                from content_analysis_agent.fewshot import load_examples
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
