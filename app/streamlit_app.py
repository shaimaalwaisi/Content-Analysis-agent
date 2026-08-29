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

from content_analysis_agent.graph import build_graph, tag_one  # noqa: E402
from content_analysis_agent.taxonomy import taxonomy_prompt  # noqa: E402
from content_analysis_agent.vlm import get_client  # noqa: E402

st.set_page_config(page_title="Content Analysis Agent", page_icon="🏷️")
st.title("🏷️ Content Analysis Agent")
st.caption("Annotate retail product images with marketing tags. Sony AI task MVP.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["anthropic", "openai", "mock"], index=0)
    model = st.text_input("Model (optional)", value="")
    context = st.text_input("Product context (optional)",
                            placeholder="Category: Mobile, Model: XPERIA10MK5")
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
            app = build_graph(client)
            with st.spinner("Tagging..."):
                tags = tag_one(app, tmp_path, context=context or None)
        except Exception as exc:
            st.error(f"Tagging failed: {exc}")
            tags = []
        finally:
            os.unlink(tmp_path)

        if tags:
            st.success("Predicted tags")
            st.write(" ".join(f"`{t}`" for t in tags))
        else:
            st.warning("No tags predicted.")

st.divider()
st.caption("Tip: set provider = mock to try the UI with no API key.")
