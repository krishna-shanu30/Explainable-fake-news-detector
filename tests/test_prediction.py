"""Tests for prediction input validation and output mapping."""
import numpy as np
import pytest

from src.prediction import Prediction, predict_text


class FakeVectorizer:
    def transform(self, texts):
        assert len(texts) == 1
        return texts


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.2, 0.8]])


def test_predict_rejects_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_text("   ")


def test_predict_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        predict_text(None)


def test_predict_returns_label_and_class_probabilities(monkeypatch):
    from src import prediction

    monkeypatch.setattr(
        prediction, "load_artifacts", lambda: (FakeModel(), FakeVectorizer())
    )
    monkeypatch.setattr(prediction, "clean_text", lambda value: value.lower())

    result = predict_text("A sample news article with enough text.")

    assert isinstance(result, Prediction)
    assert result.label == "REAL"
    assert result.confidence == pytest.approx(0.8)
    assert result.probabilities == {"FAKE": pytest.approx(0.2), "REAL": pytest.approx(0.8)}


def test_predict_rejects_text_removed_by_preprocessing(monkeypatch):
    from src import prediction

    monkeypatch.setattr(
        prediction, "load_artifacts", lambda: (FakeModel(), FakeVectorizer())
    )
    monkeypatch.setattr(prediction, "clean_text", lambda value: "")

    with pytest.raises(ValueError, match="no usable"):
        predict_text("!!! ???")
