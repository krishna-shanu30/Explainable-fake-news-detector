"""LIME explanations for individual article predictions."""
from __future__ import annotations

from functools import lru_cache

from lime.lime_text import LimeTextExplainer

from src.prediction import predict_proba


@lru_cache(maxsize=1)
def _explainer():
    return LimeTextExplainer(class_names=["FAKE", "REAL"], random_state=42)


def explain_text(text: str, num_features: int = 10) -> str:
    """Return a self-contained HTML explanation for a single article."""
    if not isinstance(text, str):
        raise TypeError("Article text must be a string.")
    if not text.strip():
        raise ValueError("Article text cannot be empty.")
    if not isinstance(num_features, int) or isinstance(num_features, bool) or num_features < 1:
        raise ValueError("num_features must be a positive integer.")

    explanation = _explainer().explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=num_features,
        labels=(0, 1),
    )
    return explanation.as_html()
