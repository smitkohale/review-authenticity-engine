"""Streamlit entry point for the Review Authenticity Engine."""

import streamlit as st

from fraudscope.config import MODEL_URL, NOTEBOOK_URL, REPO_URL
from fraudscope.ui import method_tab, rings_tab, score_tab
from fraudscope.ui.style import CSS

st.set_page_config(page_title="Review Authenticity Engine", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-kicker">Fake review detection, research demo</div>
      <h1>Review Authenticity Engine</h1>
      <p>Scores single product reviews with a fine-tuned DistilBERT classifier, and looks for groups of
      reviewer accounts acting together using a behaviour graph. Built on Amazon Electronics reviews with
      planted synthetic fakes.</p>
      <div class="app-links">
        <a href="{REPO_URL}" target="_blank">Source code</a>
        <a href="{MODEL_URL}" target="_blank">Model on Hugging Face</a>
        <a href="{NOTEBOOK_URL}" target="_blank">Training notebook</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

score, rings, method = st.tabs(["Score a review", "Coordinated clusters", "Method and results"])
with score:
    score_tab.render()
with rings:
    rings_tab.render()
with method:
    method_tab.render()
