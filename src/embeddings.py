"""Sentence-Transformer embedding features (the v2 upgrade).

Why this exists: TF-IDF matches *words*, not meaning, so it struggles on noisy,
context-dependent text (see notebooks/02 — GoEmotions tanked the bag-of-words
model). Sentence embeddings encode each message as a dense vector that captures
*semantic* meaning, so "I'm at the end of my rope" and "I can't cope" land near
each other even with no shared words.

We wrap the model in a scikit-learn transformer so it drops straight into the
existing Pipeline, and we cache encodings so repeated runs are fast.

Model: all-MiniLM-L6-v2 — small (~80MB), fast, 384-dim, a strong general-purpose
default. No raw-text preprocessing needed; transformers prefer the original text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """sklearn transformer that turns a list of texts into embedding vectors.

    Lazily loads the model on first use so importing this file is cheap and the
    saved pipeline stays lightweight (the model is re-loaded from cache, not
    pickled into the .joblib).
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _lazy_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, X, y=None):
        return self  # nothing to learn; encoding is deterministic

    def transform(self, X):
        model = self._lazy_model()
        return np.asarray(
            model.encode(
                list(X),
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # cosine-friendly; helps linear models
            )
        )

    # Keep the model out of the pickle; reload by name on unpickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state
