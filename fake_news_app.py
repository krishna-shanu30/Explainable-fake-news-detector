import streamlit as st
import joblib
import numpy as np
import re
import textstat

from scipy.sparse import hstack

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from lime.lime_text import LimeTextExplainer

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Explainable Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load("fake_news_model.pkl")

vectorizer = joblib.load("tfidf_vector.pkl")

# -------------------------------
# NLP SETUP
# -------------------------------

try:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

except LookupError:

    import nltk

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

# -------------------------------
# LIME EXPLAINER
# -------------------------------

explainer = LimeTextExplainer(
    class_names=["FAKE", "REAL"]
)

# -------------------------------
# CLEAN TEXT
# -------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# FEATURES
# -------------------------------

def read_score(text):
    return textstat.flesch_reading_ease(text)

def exclam_count(text):
    return text.count("!")

def capital_ratio(text):

    if len(text) == 0:
        return 0

    capitals = sum(1 for c in text if c.isupper())

    return capitals / len(text)

def avg_word_length(text):

    words = text.split()

    if len(words) == 0:
        return 0

    return np.mean([len(word) for word in words])

# -------------------------------
# PREDICT PROBA FOR LIME
# -------------------------------

def predict_proba(texts):

    cleaned = [clean_text(text) for text in texts]

    vect = vectorizer.transform(cleaned)

    linguistic_data = []

    for text in texts:

        readability = read_score(text)

        exclamations = exclam_count(text)

        capitals = capital_ratio(text)

        avg_length = avg_word_length(text)

        linguistic_data.append([
            readability,
            exclamations,
            capitals,
            avg_length
        ])

    linguistic_features = np.array(linguistic_data)

    final_features = hstack([
        vect,
        linguistic_features
    ])

    return model.predict_proba(final_features)

# -------------------------------
# MAIN UI
# -------------------------------

st.title("📰 Explainable Fake News Detector")

st.write(
    "Detect whether a news article is REAL or FAKE using NLP and Explainable AI."
)

news = st.text_area(
    "Paste News Article",
    height=250
)

# -------------------------------
# DETECT BUTTON
# -------------------------------

if st.button("Detect News"):

    if news.strip() == "":

        st.warning("Please enter news text.")

    else:

        cleaned = clean_text(news)

        vect = vectorizer.transform([cleaned])

        readability = read_score(news)

        exclamations = exclam_count(news)

        capitals = capital_ratio(news)

        avg_length = avg_word_length(news)

        linguistic_features = np.array([
            [
                readability,
                exclamations,
                capitals,
                avg_length
            ]
        ])

        final_features = hstack([
            vect,
            linguistic_features
        ])

        prediction = model.predict(final_features)[0]

        confidence = model.predict_proba(
            final_features
        ).max()

        # -------------------------------
        # RESULT
        # -------------------------------

        st.subheader("Prediction")

        if prediction == 0:

            st.error("⚠ FAKE NEWS DETECTED")

            label = "FAKE"

        else:

            st.success("✅ REAL NEWS")

            label = "REAL"

        st.write(f"### Label: {label}")

        st.write(
            f"### Confidence: {round(confidence * 100, 2)}%"
        )

        st.progress(float(confidence))

        # -------------------------------
        # LIME EXPLANATION
        # -------------------------------

        st.subheader("LIME Explanation")

        exp = explainer.explain_instance(
            news,
            predict_proba,
            num_features=10
        )

        html = exp.as_html()

        st.components.v1.html(
            html,
            height=800,
            scrolling=True
        )

# -------------------------------
# FOOTER
# -------------------------------

st.markdown("---")

st.markdown(
    "Built using FastAPI, Streamlit, Random Forest, TF-IDF, and LIME Explainability."
)