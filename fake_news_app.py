from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re
from scipy.sparse import hstack
import textstat
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

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

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vector.pkl")

# Input schema
class NewsInput(BaseModel):
    text: str


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


def read_score(text: str) -> float:
    return textstat.flesch_reading_ease(text)


def exclam_count(text: str) -> int:
    return text.count("!")


def capital_ratio(text: str) -> float:
    if len(text) == 0:
        return 0.0
    capitals = sum(1 for c in text if c.isupper())
    return capitals / len(text)


def avg_word_length(text: str) -> float:
    words = text.split()
    if len(words) == 0:
        return 0.0
    return float(np.mean([len(word) for word in words]))


# Prediction Route
@app.post("/predict")
def predict(data: NewsInput):

    text = data.text

    cleaned = clean_text(text)

    vect = vectorizer.transform([cleaned])

    readability = read_score(text)
    exclamations = exclam_count(text)
    capitals = capital_ratio(text)
    avg_length = avg_word_length(text)

    linguistic_features = np.array([
        [readability, exclamations, capitals, avg_length]
    ])

    final_features = hstack([vect, linguistic_features])

    prediction = model.predict(final_features)[0]

    probability = model.predict_proba(final_features).max()

    if prediction == 0:
        label = "FAKE"
    else:
        label = "REAL"

    return {
        "prediction": label,
        "confidence": float(probability)
    }
