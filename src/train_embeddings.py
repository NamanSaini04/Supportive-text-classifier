"""Train + evaluate the embedding model (v2) and compare to the TF-IDF baseline.

Pipeline: SentenceEmbedder (all-MiniLM-L6-v2) -> Logistic Regression.

Key difference from the TF-IDF path: embeddings are fed the RAW text (no
lower-casing / stop-word removal / lemmatization), because the transformer was
trained on natural text and encodes context itself.

Run:
    python -m src.train_embeddings                       # dair-ai data in data/processed
    python -m src.train_embeddings --train data/processed/train.csv \
                                   --test  data/processed/test.csv
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS_DIR, RANDOM_STATE  # noqa: E402
from src.embeddings import DEFAULT_EMBED_MODEL, SentenceEmbedder  # noqa: E402

# Recorded TF-IDF baseline (held-out, from src/evaluate.py) for a direct compare.
TFIDF_BASELINE_MACRO_F1 = 0.866


def build_head(clf_name: str):
    """The classifier on top of the embeddings: linear vs a small MLP.

    A frozen embedding + LINEAR head is a weak config; an MLP gives the dense
    semantic features a fair, non-linear chance to separate the tones.
    """
    if clf_name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=RANDOM_STATE)
    if clf_name == "mlp":
        # early_stopping left off: in some sklearn versions its validation
        # scoring path breaks on string labels. Training-loss convergence
        # (n_iter_no_change + tol) still stops us early enough.
        return MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400,
                             n_iter_no_change=15, random_state=RANDOM_STATE)
    raise ValueError(f"clf must be 'logreg' or 'mlp', got {clf_name!r}")


def build_pipeline(model_name: str = DEFAULT_EMBED_MODEL,
                   clf_name: str = "logreg") -> Pipeline:
    return Pipeline([
        ("embed", SentenceEmbedder(model_name)),
        ("clf", build_head(clf_name)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--clf", default="logreg", choices=["logreg", "mlp"],
                        help="classifier head on top of the embeddings")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Embedding model: {args.model} | head: {args.clf}\n")

    pipe = build_pipeline(args.model, args.clf)

    t0 = time.time()
    pipe.fit(train_df["text"].astype(str), train_df["label"])   # RAW text
    print(f"Fit (encode + train) took {time.time() - t0:.1f}s\n")

    y_pred = pipe.predict(test_df["text"].astype(str))
    macro = f1_score(test_df["label"], y_pred, average="macro")

    print(f"=== Held-out evaluation: embeddings + {args.clf} ===")
    print(f"Macro-F1: {macro:.3f}\n")
    print(classification_report(test_df["label"], y_pred, zero_division=0))

    delta = macro - TFIDF_BASELINE_MACRO_F1
    arrow = "up" if delta >= 0 else "down"
    print(f"TF-IDF baseline macro-F1: {TFIDF_BASELINE_MACRO_F1:.3f}")
    print(f"Embeddings macro-F1:      {macro:.3f}  ({arrow} {abs(delta):.3f})")

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / f"embed_{args.clf}_pipeline.joblib"
    joblib.dump(pipe, out)
    print(f"\nSaved model -> {out}  (embedder reloaded by name, not pickled)")


if __name__ == "__main__":
    main()
