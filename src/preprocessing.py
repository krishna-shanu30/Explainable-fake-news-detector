"""Text preprocessing shared by model inference and LIME."""
from __future__ import annotations

import re
from functools import lru_cache

_URL_RE = re.compile(r"http\S+", flags=re.IGNORECASE)
_NON_ALPHANUMERIC_RE = re.compile(r"[^a-zA-Z0-9\s]+")


@lru_cache(maxsize=1)
def _nltk_resources():
    """Load the English resources used by the original training script."""
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("articles")
        return words, lemmatizer
    except LookupError as exc:
        raise RuntimeError(
            "NLTK data is missing. Run: python -m nltk.downloader stopwords wordnet"
        ) from exc


def clean_text(text: str) -> str:
    """Match the original model-training preprocessing as closely as possible."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = _URL_RE.sub("", text.lower())
    text = _NON_ALPHANUMERIC_RE.sub("", text)
    stop_words, lemmatizer = _nltk_resources()
    tokens = [
        lemmatizer.lemmatize(token)
        for token in text.split()
        if token not in stop_words
    ]
    return " ".join(tokens)
