"""Tests for input validation and text preprocessing."""
import pytest

from src.preprocessing import clean_text


def test_clean_text_rejects_non_string():
    with pytest.raises(TypeError):
        clean_text(None)


def test_clean_text_rejects_number():
    with pytest.raises(TypeError):
        clean_text(123)


def test_clean_text_returns_string():
    # This test also checks that the configured NLTK resources are available.
    result = clean_text("The quick brown foxes are running!")
    assert isinstance(result, str)
    assert "the" not in result.split()
