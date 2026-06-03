"""Remap fine-grained public-dataset labels into our six tones.

Real datasets (GoEmotions, dair-ai/emotion) use their own label sets. Map them
onto config.TONE_LABELS here, then write the result to data/processed/train.csv
with columns: text,label.

'loneliness' rarely appears as a clean label in public data — derive it from
sadness samples that contain loneliness keywords (see derive_loneliness).
"""
from config import TONE_LABELS  # noqa: F401  (import to keep label source single)

GOEMOTIONS_TO_TONE = {
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
    "anger": "anger", "annoyance": "anger",
    "nervousness": "anxiety", "fear": "anxiety",
    "neutral": "neutral",
    # stress is not a native GoEmotions label; derive from keywords if needed
}

DAIR_AI_TO_TONE = {
    "sadness": "sadness",
    "anger": "anger",
    "fear": "anxiety",
    "joy": "neutral",      # collapse positive into neutral for THIS app's scope
    "love": "neutral",
    "surprise": "neutral",
}

_LONELINESS_KEYWORDS = ("alone", "lonely", "no one", "nobody", "isolated", "left out")
_STRESS_KEYWORDS = ("overwhelmed", "too much", "deadline", "pressure", "exhausted")


def derive_loneliness(text: str, current_label: str) -> str:
    """Promote a sadness sample to loneliness when keywords are present."""
    if current_label == "sadness" and any(k in text.lower() for k in _LONELINESS_KEYWORDS):
        return "loneliness"
    return current_label


def derive_stress(text: str, current_label: str) -> str:
    if current_label in ("anxiety", "neutral") and any(
        k in text.lower() for k in _STRESS_KEYWORDS
    ):
        return "stress"
    return current_label
