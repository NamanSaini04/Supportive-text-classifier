"""Text preprocessing pipeline for the TF-IDF baseline.

Each step is chosen deliberately:
- lowercasing        -> shrink vocabulary ("Sad" and "sad" become one feature)
- URL / @mention strip -> remove noise AND anonymize handles
- keep ! and ?       -> punctuation carries emotional intensity / uncertainty
- KEEP negation words -> "not okay" must not become "okay" (most important choice)
- lemmatize          -> "worried" -> "worry"; real words, better interpretability

NOTE: This heavy cleaning is for TF-IDF. If you upgrade to Sentence
Transformers later, you would feed raw text instead.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk():
    """Download required NLTK data once, quietly."""
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()

_LEMMATIZER = WordNetLemmatizer()

# Remove generic stopwords BUT keep negations — they flip emotional meaning.
_STOP = set(stopwords.words("english")) - {"not", "no", "nor", "never", "n't"}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions, keep emotional punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs carry no tone
    text = re.sub(r"@\w+", " ", text)               # anonymize handles
    text = re.sub(r"[^a-z\s!?']", " ", text)        # keep ! ? ' for signal
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list[str]:
    tokens = text.split()
    return [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP and len(tok) > 1
    ]


def preprocess(text: str) -> str:
    """Full pipeline -> space-joined string ready for TF-IDF."""
    return " ".join(tokenize_and_lemmatize(clean_text(text)))


if __name__ == "__main__":
    samples = [
        "I'm NOT okay, everything is falling apart...",
        "Check out http://example.com @friend it's fine!",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {preprocess(s)!r}\n")
