"""Model training workflow: TF-IDF + a linear classifier.

Why TF-IDF + Logistic Regression as the baseline?
  - TF-IDF weights words by how distinctive they are (common words down,
    rare/meaningful words up).
  - Logistic Regression is fast, interpretable, and gives predict_proba ->
    we can show a real confidence score in the UI.
  - ALWAYS baseline first; only reach for SVM / transformers if this isn't
    good enough.

Run:  python -m src.train --data data/processed/train.csv --model logreg
"""
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import DEFAULT_MODEL, MODELS_DIR, RANDOM_STATE
from src.preprocessing import preprocess


def load_data(path: str) -> pd.DataFrame:
    """Expects a CSV with columns: text, label."""
    df = pd.read_csv(path)
    df["clean"] = df["text"].astype(str).apply(preprocess)
    return df


def build_pipeline(model_name: str = DEFAULT_MODEL) -> Pipeline:
    models = {
        # class_weight balances out the dominant "neutral" class
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "nb": MultinomialNB(),
        "svm": LinearSVC(class_weight="balanced"),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list(models)}")
    return Pipeline([
        # ngram (1,2) captures phrases like "not okay", "shut down"
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("clf", models[model_name]),
    ])


def train(data_path: str, model_name: str = DEFAULT_MODEL) -> Pipeline:
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.2,
        stratify=df["label"],          # keep class balance honest
        random_state=RANDOM_STATE,
    )

    pipe = build_pipeline(model_name)

    # Macro-F1 weights every emotion equally — the metric we actually care about
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"[{model_name}] CV macro-F1: {cv.mean():.3f} (+/- {cv.std():.3f})")

    pipe.fit(X_train, y_train)
    print("\n--- Held-out test report ---")
    print(classification_report(y_test, pipe.predict(X_test)))

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / f"{model_name}_pipeline.joblib"
    joblib.dump(pipe, out)
    print(f"Saved model -> {out}")
    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["logreg", "nb", "svm"])
    args = parser.parse_args()
    train(args.data, args.model)
