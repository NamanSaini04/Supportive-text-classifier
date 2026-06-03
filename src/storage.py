"""Anonymized local logging for trend tracking.

IMPORTANT: we store ONLY the derived signals (tone, confidence, distress flag,
timestamp). We NEVER store the user's raw text. That is what "anonymized"
must actually mean here.
"""
import csv
from datetime import datetime, timezone

import pandas as pd

from config import LOG_PATH

_FIELDS = ["timestamp", "tone", "confidence", "urgent_distress"]


def log_entry(result: dict) -> None:
    """Append one derived-signal row. No raw text is ever written."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tone": result["tone"],
        "confidence": result["confidence"],
        "urgent_distress": result["urgent_distress"],
    }
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_log() -> pd.DataFrame:
    """Load the log for analytics. Returns empty DataFrame if none yet."""
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=_FIELDS)
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df


def tone_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Tone counts per day — ready to chart."""
    if df.empty:
        return df
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    return df.groupby(["date", "tone"]).size().unstack(fill_value=0)
