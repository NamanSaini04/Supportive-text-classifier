"""Supportive, NON-CLINICAL response engine.

The wording here IS the ethics of this project. Rules:
  - validate feelings, never diagnose ("this sounds anxious", not "you have anxiety")
  - never prescribe treatment
  - if the distress flag fires, crisis resources OVERRIDE everything else
"""
from config import CRISIS_MESSAGE, DISCLAIMER  # noqa: F401  (re-exported for app)

SUPPORTIVE = {
    "neutral": (
        "Thanks for sharing. Putting your thoughts into words is a healthy habit."
    ),
    "stress": (
        "That sounds like a lot to carry right now. A short break or a few slow "
        "breaths might help — be gentle with yourself."
    ),
    "anxiety": (
        "It sounds like things feel uncertain or overwhelming. Naming what you "
        "feel, like you just did, is a real step."
    ),
    "sadness": (
        "I'm sorry things feel heavy. Your feelings are valid, and reaching out "
        "to someone you trust can help."
    ),
    "anger": (
        "It sounds like something really frustrated you. Letting it out in words "
        "is a constructive outlet."
    ),
    "loneliness": (
        "Feeling disconnected is hard. You're not alone in feeling this way — "
        "connecting with one person today, even briefly, can matter."
    ),
}


def supportive_response(tone: str, urgent: bool) -> str:
    """Return the message to show the user. Crisis path overrides tone."""
    if urgent:
        return CRISIS_MESSAGE
    return SUPPORTIVE.get(tone, SUPPORTIVE["neutral"])
