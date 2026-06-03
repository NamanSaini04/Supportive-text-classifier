"""Download the dair-ai/emotion dataset and remap it into our six tones.

dair-ai/emotion ships 6 labels (integers):
    0 sadness | 1 joy | 2 love | 3 anger | 4 fear | 5 surprise

We map those onto config.TONE_LABELS via data/label_map.py:
    sadness -> sadness | anger -> anger | fear -> anxiety
    joy/love/surprise -> neutral   (this app only scopes negative/neutral tone)
then DERIVE the two tones the source lacks:
    loneliness  (from sadness samples with loneliness keywords)
    stress      (from anxiety/neutral samples with stress keywords)

This is an honest, documented remap — say so in your README. The source has no
native loneliness/stress label, so those classes are keyword-derived and will be
noisier; that is a known limitation, not a bug.

Usage:
    python -m src.download_data                 # full train split -> train.csv
    python -m src.download_data --max-per-class 2000   # cap for speed/balance
    python -m src.download_data --split validation
"""
import sys
from pathlib import Path

import pandas as pd

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_map import DAIR_AI_TO_TONE, derive_loneliness, derive_stress  # noqa: E402
from src.dataset_utils import run_cli  # noqa: E402

# Parquet files served directly by the Hugging Face Hub (no auth needed)
_BASE = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split"
_FILES = {
    "train": f"{_BASE}/train-00000-of-00001.parquet",
    "validation": f"{_BASE}/validation-00000-of-00001.parquet",
    "test": f"{_BASE}/test-00000-of-00001.parquet",
}

# dair-ai/emotion integer label -> source emotion name
_INT_TO_EMOTION = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def fetch_split(split: str) -> pd.DataFrame:
    """Download one split's parquet into a DataFrame with text + emotion name."""
    if split not in _FILES:
        raise ValueError(f"split must be one of {list(_FILES)}")
    print(f"Downloading dair-ai/emotion [{split}] ...")
    df = pd.read_parquet(_FILES[split])          # pyarrow handles the HTTP+parquet
    df = df.rename(columns={"label": "label_int"})
    df["emotion"] = df["label_int"].map(_INT_TO_EMOTION)
    print(f"  got {len(df)} rows")
    return df[["text", "emotion"]]


def remap_to_tones(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the base map, then derive loneliness and stress from keywords."""
    df = df.copy()
    df["label"] = df["emotion"].map(DAIR_AI_TO_TONE)
    df = df.dropna(subset=["label"])             # drop anything unmapped

    # Order matters: derive the missing classes from the mapped ones.
    df["label"] = [derive_loneliness(t, l) for t, l in zip(df["text"], df["label"])]
    df["label"] = [derive_stress(t, l) for t, l in zip(df["text"], df["label"])]
    return df[["text", "label"]]


if __name__ == "__main__":
    run_cli(fetch_split, remap_to_tones, list(_FILES))
