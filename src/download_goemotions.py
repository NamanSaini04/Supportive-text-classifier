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
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_PROCESSED  # noqa: E402
from data.label_map import derive_loneliness, derive_stress  # noqa: E402
from src.dataset_utils import build_and_save  # noqa: E402

_BASE = "https://huggingface.co/api/datasets/google-research-datasets/go_emotions/parquet/simplified"
_FILES = {
    "train": f"{_BASE}/train/0.parquet",
    "validation": f"{_BASE}/validation/0.parquet",
    "test": f"{_BASE}/test/0.parquet",
}

# GoEmotions integer id -> our tone. Note id 27 (true neutral) maps to neutral.
# The 18 positive/ambiguous emotions (admiration, amusement, approval, caring,
# confusion, curiosity, desire, embarrassment, excitement, gratitude, joy, love,
# optimism, pride, realization, relief, surprise) are intentionally NOT here.
_GOEMOTIONS_TO_TONE = {
    2: "anger", 3: "anger", 10: "anger", 11: "anger",          # anger, annoyance, disapproval, disgust
    14: "anxiety", 19: "anxiety",                              # fear, nervousness
    9: "sadness", 16: "sadness", 24: "sadness", 25: "sadness",  # disappointment, grief, remorse, sadness
    27: "neutral",                                            # true neutral only
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


def remap_to_tones(df: pd.DataFrame, strict_neutral: bool = True) -> pd.DataFrame:
    """Map GoEmotions ids onto our six tones.

    strict_neutral=True (the fix): unmapped ids = the 18 positive/ambiguous
        emotions -> DROPPED, so `neutral` only contains true-neutral text.
    strict_neutral=False (old behaviour): unmapped -> folded into `neutral`,
        creating the oversized grab-bag that tanked the original experiment.
    """
    df = df.copy()
    df["label"] = df["label_int"].map(_GOEMOTIONS_TO_TONE)
    if strict_neutral:
        df = df.dropna(subset=["label"])          # drop positives, don't mislabel
    else:
        df["label"] = df["label"].fillna("neutral")
    # derive the two tones GoEmotions lacks
    df["label"] = [derive_loneliness(t, l) for t, l in zip(df["text"], df["label"])]
    df["label"] = [derive_stress(t, l) for t, l in zip(df["text"], df["label"])]
    return df[["text", "label"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=list(_FILES))
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--lenient-neutral", action="store_true",
                        help="fold ALL positive emotions into neutral (old, broken scheme)")
    args = parser.parse_args()

    remap = partial(remap_to_tones, strict_neutral=not args.lenient_neutral)
    out = Path(args.out) if args.out else DATA_PROCESSED / f"{args.split}.csv"
    build_and_save(fetch_split, remap, args.split, args.max_per_class, out)


if __name__ == "__main__":
    main()
