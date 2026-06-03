"""Download the GoEmotions dataset (Google) and remap it into our six tones.

GoEmotions has 28 fine-grained labels (27 emotions + neutral) on ~43k Reddit
comments. It is multi-label, so we keep only single-label examples (~36k) for a
clean classification target, then collapse the 28 labels onto our six tones.

Why GoEmotions: it carries richer NEGATIVE emotions than dair-ai/emotion
(grief, disappointment, remorse, nervousness, disgust), giving the model more and
more varied data for sadness/anxiety/anger. It still has no native stress or
loneliness label, so those two remain keyword-derived (a documented limitation).

Usage:
    python -m src.download_goemotions                      # train -> train.csv
    python -m src.download_goemotions --split test         # -> test.csv
    python -m src.download_goemotions --max-per-class 2500  # cap for balance
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_map import derive_loneliness, derive_stress  # noqa: E402
from src.dataset_utils import run_cli  # noqa: E402

_BASE = "https://huggingface.co/api/datasets/google-research-datasets/go_emotions/parquet/simplified"
_FILES = {
    "train": f"{_BASE}/train/0.parquet",
    "validation": f"{_BASE}/validation/0.parquet",
    "test": f"{_BASE}/test/0.parquet",
}

# GoEmotions integer id -> our tone. Positives and ambiguous emotions -> neutral.
_GOEMOTIONS_TO_TONE = {
    2: "anger", 3: "anger", 10: "anger", 11: "anger",        # anger, annoyance, disapproval, disgust
    14: "anxiety", 19: "anxiety",                            # fear, nervousness
    9: "sadness", 16: "sadness", 24: "sadness", 25: "sadness",  # disappointment, grief, remorse, sadness
    # everything else (admiration, amusement, approval, caring, confusion,
    # curiosity, desire, embarrassment, excitement, gratitude, joy, love,
    # optimism, pride, realization, relief, surprise, neutral) -> neutral
}


def fetch_split(split: str) -> pd.DataFrame:
    if split not in _FILES:
        raise ValueError(f"split must be one of {list(_FILES)}")
    print(f"Downloading GoEmotions [{split}] ...")
    df = pd.read_parquet(_FILES[split])
    df["n"] = df["labels"].apply(len)
    single = df[df["n"] == 1].copy()                    # keep single-label only
    single["label_int"] = single["labels"].apply(lambda a: int(a[0]))
    print(f"  {len(df)} rows, {len(single)} single-label kept")
    return single[["text", "label_int"]]


def remap_to_tones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # default everything to neutral, then override the negative ids
    df["label"] = df["label_int"].map(_GOEMOTIONS_TO_TONE).fillna("neutral")
    # derive the two tones GoEmotions lacks
    df["label"] = [derive_loneliness(t, l) for t, l in zip(df["text"], df["label"])]
    df["label"] = [derive_stress(t, l) for t, l in zip(df["text"], df["label"])]
    return df[["text", "label"]]


if __name__ == "__main__":
    run_cli(fetch_split, remap_to_tones, list(_FILES))
