"""Text preprocessing shared by model inference and LIME."""
from __future__ import annotations

import re
from functools import lru_cache

_URL_RE = re.compile(r"http\S+", flags=re.IGNORECASE)
_NON_ALPHANUMERIC_RE = re.compile(r"[^a-zA-Z0-9\s]+")


@lru_cache(maxsize=1)
def _nltk_resources():
    """Load required English corpora, downloading them on first cloud startup if needed."""
    import nltk

    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("articles")
        return words, lemmatizer
    except LookupError:
        # Streamlit Community Cloud does not automatically install NLTK corpora
        # when pip installs requirements.txt. Fetch only the two resources used.
        for resource in ("stopwords", "wordnet"):
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(
                    "Could not download the NLTK language resources. "
                    "Check the app logs and redeploy."
                )
        try:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer

            words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("articles")
            return words, lemmatizer
        except LookupError as exc:
            raise RuntimeError(
                "NLTK language resources are still unavailable after download. "
                "Check the app logs and redeploy."
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
