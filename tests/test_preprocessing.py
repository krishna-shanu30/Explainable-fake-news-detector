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
    result = clean_text("The quick brown foxes are running!")
    assert isinstance(result, str)
    assert "the" not in result.split()


def test_clean_text_removes_urls_and_punctuation():
    result = clean_text("Visit https://example.com/path now! Breaking-news, today.")
    assert "https" not in result
    assert "example" not in result
    assert "!" not in result
    assert "," not in result
    assert "-" not in result
    assert "breakingnews" in result
