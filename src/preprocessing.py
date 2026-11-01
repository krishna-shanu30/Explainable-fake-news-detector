"""Text preprocessing shared by model inference and LIME."""
from __future__ import annotations

import re
from functools import lru_cache

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]+")
_SPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _nltk_resources():
    """Load the same English stopwords and lemmatizer used by training."""
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        # Trigger a resource check now rather than during a prediction.
        lemmatizer.lemmatize("articles")
        return words, lemmatizer
    except LookupError as exc:
        raise RuntimeError(
            "NLTK data is missing. Run: python -m nltk.downloader stopwords wordnet omw-1.4"
        ) from exc


def clean_text(text: str) -> str:
    """Normalize article text using the training script's preprocessing steps."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = _URL_RE.sub(" ", text.lower())
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    stop_words, lemmatizer = _nltk_resources()

    tokens = [
        lemmatizer.lemmatize(token)
        for token in text.split()
        if token not in stop_words
    ]
    return " ".join(tokens)
