"""Inference: combine the emotion model with the distress safety layer.

Two systems run in parallel and are combined here:
  1. emotion model  -> tone + confidence (on cleaned text)
  2. distress layer -> urgent flag       (on RAW text)

The distress flag is a SEPARATE field and is never overwritten by the model.
"""
import functools

import joblib
import numpy as np

from config import DEFAULT_MODEL, MODELS_DIR
from src.distress import check_distress
from src.preprocessing import preprocess


@functools.lru_cache(maxsize=1)
def _load_pipeline(model_name: str = DEFAULT_MODEL):
    path = MODELS_DIR / f"{model_name}_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python -m src.train first."
        )
    return joblib.load(path)


def classify(raw_text: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Classify one message. Returns tone, confidence, all scores, distress flag."""
    # 1) Safety FIRST, on raw text (before any word stripping)
    distress = check_distress(raw_text)

    pipe = _load_pipeline(model_name)
    clean = preprocess(raw_text)
    proba = pipe.predict_proba([clean])[0]
    idx = int(np.argmax(proba))

    return {
        "tone": str(pipe.classes_[idx]),
        "confidence": round(float(proba[idx]), 3),
        "all_scores": {
            str(c): round(float(p), 3) for c, p in zip(pipe.classes_, proba)
        },
        "urgent_distress": distress,   # separate, never folded into the tone
    }


if __name__ == "__main__":
    import json
    print(json.dumps(classify("I feel so alone and nobody understands me"), indent=2))
