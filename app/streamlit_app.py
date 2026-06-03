"""Streamlit UI — single, calm screen with the disclaimer always visible.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from config import DISCLAIMER  # noqa: E402
from src.predict import classify  # noqa: E402
from src.responses import supportive_response  # noqa: E402
from src.storage import load_log, log_entry, tone_trends  # noqa: E402

st.set_page_config(page_title="Emotion Reflection", page_icon="🌱")
st.title("🌱 Emotion & Tone Reflection")
st.caption(DISCLAIMER)  # disclaimer ALWAYS on screen

text = st.text_area("How are you feeling? Write a sentence or two:", height=140)

if st.button("Reflect") and text.strip():
    try:
        result = classify(text)
    except FileNotFoundError as e:
        st.warning(str(e))
    else:
        if result["urgent_distress"]:
            # Crisis path FIRST and visually distinct (red).
            st.error(supportive_response(result["tone"], urgent=True))
        else:
            st.subheader(f"Tone: {result['tone'].title()}")
            st.progress(result["confidence"])
            st.caption(f"Confidence: {result['confidence']:.0%}")
            st.info(supportive_response(result["tone"], urgent=False))
            # Showing ALL class scores is honest about the model's uncertainty
            st.bar_chart(result["all_scores"])

        log_entry(result)  # anonymized: derived signals only

with st.expander("📊 My reflection trends"):
    trends = tone_trends(load_log())
    if trends.empty:
        st.write("No entries logged yet. Reflect on a message to start tracking.")
    else:
        st.line_chart(trends)
        st.caption(
            "These trends reflect the tone of your writing over time. "
            "Patterns here are for personal reflection only — they are not a "
            "clinical assessment."
        )
