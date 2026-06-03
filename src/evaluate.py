"""Evaluate the trained model on a held-out test set + check the distress layer.

The test split comes from dair-ai/emotion's own test partition — data the model
never saw during training, so these numbers reflect real generalization (unlike
the in-script train/test split inside src/train.py).

Outputs:
  - classification report (per-class precision/recall/F1) to stdout
  - a confusion-matrix PNG to reports/confusion_matrix.png
  - the distress-layer recall on a small labeled probe set

Usage:
    python -m src.evaluate                       # uses data/processed/test.csv
    python -m src.evaluate --data path/to.csv --model logreg
"""
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed; we save to file
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_MODEL, MODELS_DIR, ROOT  # noqa: E402
from src.distress import check_distress  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402

REPORTS = ROOT / "reports"

# Small labeled probe set for the SAFETY layer. We report recall here separately
# because a missed true-distress message is the costliest error in this project.
_DISTRESS_PROBE = [
    ("i don't want to be here anymore", True),
    ("i can't take it anymore", True),
    ("sometimes i just want to disappear", True),
    ("everyone would be better off without me", True),
    ("i want to die", True),
    ("i'm so tired of this rainy weather", False),
    ("work was stressful but i'm okay", False),
    ("i feel a bit lonely tonight", False),
    ("i'm frustrated with my group project", False),
    ("just a normal boring day", False),
]


def evaluate_model(data_path: str, model_name: str):
    model_file = MODELS_DIR / f"{model_name}_pipeline.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"No model at {model_file}. Run python -m src.train.")
    pipe = joblib.load(model_file)

    df = pd.read_csv(data_path)
    df["clean"] = df["text"].astype(str).apply(preprocess)
    y_true = df["label"]
    y_pred = pipe.predict(df["clean"])

    labels = sorted(y_true.unique())
    print(f"\n=== Held-out evaluation: {model_name} on {data_path} ({len(df)} rows) ===")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.3f}\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"Confusion matrix — {model_name} (held-out test)")
    plt.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved confusion matrix -> {out}")


def evaluate_distress():
    print("\n=== Distress safety layer (reported separately) ===")
    tp = fn = fp = tn = 0
    for text, is_distress in _DISTRESS_PROBE:
        flagged = check_distress(text)
        if is_distress and flagged:
            tp += 1
        elif is_distress and not flagged:
            fn += 1
            print(f"  MISSED (false negative): {text!r}")
        elif not is_distress and flagged:
            fp += 1
            print(f"  False alarm (acceptable): {text!r}")
        else:
            tn += 1
    pos = tp + fn
    recall = tp / pos if pos else float("nan")
    print(f"  Recall on probe set: {recall:.2f}  (TP={tp}, FN={fn}, FP={fp}, TN={tn})")
    print("  Note: tuned for recall by design; false positives are acceptable here.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/test.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    evaluate_model(args.data, args.model)
    evaluate_distress()


if __name__ == "__main__":
    main()
