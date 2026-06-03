"""Shared dataset-building helpers used by the dataset downloaders.

Both `src/download_data.py` (dair-ai/emotion) and `src/download_goemotions.py`
follow the same shape: fetch a split, remap its labels onto our six tones,
optionally cap per class, report the distribution, and save a CSV. Only the
`fetch` and `remap` steps differ per dataset, so they live in the dataset
modules; everything else is shared here.
"""
import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_PROCESSED, RANDOM_STATE, TONE_LABELS  # noqa: E402

# A fetch returns a raw DataFrame; a remap turns it into columns [text, label].
FetchFn = Callable[[str], pd.DataFrame]
RemapFn = Callable[[pd.DataFrame], pd.DataFrame]


def balance(df: pd.DataFrame, max_per_class: int | None,
            random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Optionally cap each class to reduce imbalance, then shuffle."""
    if not max_per_class:
        return df
    parts = [
        g.sample(min(len(g), max_per_class), random_state=random_state)
        for _, g in df.groupby("label")
    ]
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def build_and_save(fetch: FetchFn, remap: RemapFn, split: str,
                   max_per_class: int | None, out_path: Path) -> pd.DataFrame:
    """Run the full fetch -> remap -> balance -> report -> save pipeline."""
    tones = balance(remap(fetch(split)), max_per_class)

    print("\nClass distribution after remap:")
    counts = tones["label"].value_counts().reindex(TONE_LABELS).fillna(0).astype(int)
    print(counts.to_string())
    missing = counts[counts == 0].index.tolist()
    if missing:
        print(f"\n  NOTE: no samples for {missing} — keyword derivation found none. "
              "Loosen keywords in data/label_map.py or add data manually.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tones.to_csv(out_path, index=False)
    print(f"\nSaved {len(tones)} rows -> {out_path}")
    return tones


def run_cli(fetch: FetchFn, remap: RemapFn, splits: list[str]) -> None:
    """Standard command-line entry point shared by both downloaders."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=splits)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="cap samples per tone (helps imbalance + speed)")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args()

    out = Path(args.out) if args.out else DATA_PROCESSED / f"{args.split}.csv"
    build_and_save(fetch, remap, args.split, args.max_per_class, out)
