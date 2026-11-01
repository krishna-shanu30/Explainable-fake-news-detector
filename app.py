"""Streamlit interface for the Explainable Fake News Detector."""
from __future__ import annotations

import streamlit as st

from src.explainability import explain_text
from src.prediction import ModelLoadError, predict_text

st.set_page_config(
    page_title="Explainable Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Explainable Fake News Detector")
st.caption("An educational NLP classifier with local explanations.")

st.info(
    "This tool estimates whether text resembles examples in its training dataset. "
    "It does not verify claims against trusted sources and must not be used as "
    "the sole basis for sharing, publishing, or acting on a news story."
)

with st.sidebar:
    st.header("About this project")
    st.write("TF-IDF + Random Forest + LIME")
    st.write("Predictions are based on the model and dataset used during training.")
    st.caption("Avoid entering private, confidential, or personally identifying information.")

news = st.text_area(
    "Paste a news article",
    height=260,
    max_chars=50_000,
    placeholder="Paste the article text here...",
)

analyze = st.button("Analyze article", type="primary", use_container_width=True)

if analyze:
    if not news.strip():
        st.warning("Please paste some article text before analyzing.")
    elif len(news.strip()) < 40:
        st.warning("Please enter a little more text (at least 40 characters) for a meaningful analysis.")
    else:
        try:
            with st.spinner("Analyzing article..."):
                result = predict_text(news)
            st.subheader("Model result")
            col1, col2 = st.columns(2)
            with col1:
                if result.label == "FAKE":
                    st.error("Model prediction: FAKE")
                else:
                    st.success("Model prediction: REAL")
            with col2:
                st.metric("Model confidence", f"{result.confidence:.1%}")
            st.progress(float(result.confidence))
            st.caption(
                "Confidence is the model's predicted probability for its selected class; "
                "it is not a calibrated probability that the article is true or false."
            )

            with st.expander("Class probabilities", expanded=True):
                st.write(
                    {
                        "FAKE": f"{result.probabilities['FAKE']:.1%}",
                        "REAL": f"{result.probabilities['REAL']:.1%}",
                    }
                )

            st.subheader("Why did the model make this prediction?")
            with st.spinner("Generating a local LIME explanation..."):
                explanation = explain_text(news)
            st.components.v1.html(explanation, height=620, scrolling=True)
        except ModelLoadError as exc:
            st.error(str(exc))
        except Exception as exc:  # Keep unexpected failures visible without crashing the page.
            st.error("The analysis could not be completed. Check the model files and application logs.")
            st.exception(exc)

st.divider()
st.caption("Built for learning and demonstration. Always verify news with independent, reliable sources.")
