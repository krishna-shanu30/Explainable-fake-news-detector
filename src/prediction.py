"""Model loading and prediction. Feature construction matches the checked-in training script."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "fake_news_model.pkl"
VECTORIZER_PATH = ROOT / "tfidf_vector.pkl"


class ModelLoadError(RuntimeError):
    """Raised when model artifacts are missing or cannot be loaded."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def load_artifacts():
    """Load trusted, repository-managed model artifacts once per process."""
    if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        raise ModelLoadError(
            "Model files are missing. Ensure fake_news_model.pkl and "
            "tfidf_vector.pkl are present in the repository root."
        )
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as exc:
        raise ModelLoadError(
            "Could not load the model artifacts. Check that they were produced "
            "with compatible scikit-learn and joblib versions."
        ) from exc

    if not hasattr(model, "predict_proba") or not hasattr(model, "predict"):
        raise ModelLoadError("The saved classifier does not support the required prediction methods.")
    if not hasattr(vectorizer, "transform"):
        raise ModelLoadError("The saved TF-IDF vectorizer is invalid.")
    return model, vectorizer


def predict_proba(texts: list[str]):
    """Return class probabilities for LIME using the exact training feature space."""
    model, vectorizer = load_artifacts()
    cleaned = [clean_text(text) for text in texts]
    features = vectorizer.transform(cleaned)
    return model.predict_proba(features)


def predict_text(text: str) -> Prediction:
    """Classify one article using the TF-IDF-only pipeline used to train the model."""
    if not isinstance(text, str):
        raise TypeError("Article text must be a string.")
    if not text.strip():
        raise ValueError("Article text cannot be empty.")

    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("The article contains no usable English text after preprocessing.")

    features = vectorizer.transform([cleaned])
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    class_to_label = {0: "FAKE", 1: "REAL"}
    if predicted_class not in class_to_label:
        raise ModelLoadError(f"Unexpected model class: {predicted_class}")

    probability_by_label = {}
    for class_value, probability in zip(model.classes_, probabilities):
        label = class_to_label.get(int(class_value))
        if label is not None:
            probability_by_label[label] = float(probability)

    label = class_to_label[predicted_class]
    return Prediction(
        label=label,
        confidence=probability_by_label[label],
        probabilities={
            "FAKE": probability_by_label.get("FAKE", 0.0),
            "REAL": probability_by_label.get("REAL", 0.0),
        },
    )
