"""Tests for LIME explanation input validation."""
import pytest

from src.explainability import explain_text


@pytest.mark.parametrize("value", [None, 123, []])
def test_explain_rejects_non_string(value):
    with pytest.raises(TypeError, match="must be a string"):
        explain_text(value)


def test_explain_rejects_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        explain_text("  ")


@pytest.mark.parametrize("num_features", [0, -1, 1.5, True])
def test_explain_rejects_invalid_feature_count(num_features):
    with pytest.raises(ValueError, match="positive integer"):
        explain_text("A sufficiently long news article.", num_features=num_features)
