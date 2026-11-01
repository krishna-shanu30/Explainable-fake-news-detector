"""Tests for prediction input validation."""
import pytest

from src.prediction import predict_text


def test_predict_rejects_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_text("   ")


def test_predict_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        predict_text(None)
