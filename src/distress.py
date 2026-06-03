"""URGENT DISTRESS SAFETY LAYER.

This is the most important module in the project. It is intentionally:
  - RULE-BASED (auditable, no black box where false negatives are dangerous)
  - HIGH-RECALL (we accept false positives to avoid missing real distress)
  - RUN ON RAW TEXT (before cleaning, so we never strip critical words)

It is deliberately NOT an ML model. A small classifier trained on scraped
crisis data would be unreliable in exactly the situation where reliability
matters most. We choose a transparent net we can reason about and test.

This is a safety net, NOT a guarantee. It will miss indirect / coded language.
"""
import re

# Curated, reviewed patterns. Tuned for recall. Add to this list carefully and
# always with a corresponding test in tests/test_distress.py.
_DISTRESS_PATTERNS = [
    r"\b(kill|hurt|harm)(ing)?\s+(myself|me)\b",
    r"\bend(ing)?\s+(it|my life|things)\b",
    r"\b(don'?t|do not)\s+want\s+to\s+(be here|live|wake up)\b",
    r"\bno\s+(reason|point)\s+(to|in)\s+(live|living|go on|going on)\b",
    r"\bcan'?t\s+(go on|do this anymore|take\s+it\s+anymore|take\s+anymore)\b",
    r"\bwant\s+to\s+(die|disappear)\b",
    r"\b(better off|everyone would be better)\s+(without me|dead)\b",
    r"\bgiving up\s+on\s+(everything|life)\b",
]

_RE = [re.compile(p, re.IGNORECASE) for p in _DISTRESS_PATTERNS]


def check_distress(raw_text: str) -> bool:
    """Return True if the RAW text contains urgent-distress language.

    Call this on the original text, BEFORE preprocessing.
    """
    if not raw_text:
        return False
    return any(rx.search(raw_text) for rx in _RE)


if __name__ == "__main__":
    for t in ["i don't want to be here anymore", "i'm just tired of the rain"]:
        print(f"{t!r} -> distress={check_distress(t)}")
