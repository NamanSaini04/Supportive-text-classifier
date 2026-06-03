"""Fine-tune the transformer (v3) — the real test of whether it can beat TF-IDF.

In notebook 03 we used FROZEN embeddings (the transformer was a fixed feature
extractor). That lost to TF-IDF. The natural follow-up: actually FINE-TUNE the
transformer end-to-end, so its weights adapt to our six-tone task.

Backbone: sentence-transformers/all-MiniLM-L6-v2 (same as the frozen experiment,
so the comparison is apples-to-apples) with a fresh classification head.

No `accelerate`/`Trainer` dependency — this is a plain PyTorch loop on Apple MPS
(falls back to CPU). Class-weighted loss mirrors `class_weight="balanced"`.

Run:
    python -m src.train_finetune                       # dair-ai data, 3 epochs
    python -m src.train_finetune --epochs 4 --batch-size 32
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS_DIR, RANDOM_STATE, TONE_LABELS  # noqa: E402

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_BASELINE = 0.866      # held-out, for a direct compare
FROZEN_EMBED_MLP = 0.673    # from notebook 03


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ToneDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = pick_device()
    print(f"Device: {device} | backbone: {BACKBONE}\n")

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Stable label <-> id mapping (use our canonical tone order)
    labels = [t for t in TONE_LABELS if t in set(train_df["label"])]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    y_train = train_df["label"].map(label2id).to_numpy()
    y_test = test_df["label"].map(label2id).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=len(labels), id2label=id2label, label2id=label2id,
    ).to(device)

    train_ds = ToneDataset(train_df["text"].astype(str), y_train, tokenizer)
    test_ds = ToneDataset(test_df["text"].astype(str), y_test, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    # Class-weighted loss == class_weight="balanced"
    cw = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_train)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | "
          f"{args.epochs} epochs, batch {args.batch_size}\n")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            loss = loss_fn(out.logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"  epoch {epoch}/{args.epochs}  avg loss {running/len(train_dl):.4f}")
    print(f"\nFine-tuning took {time.time() - t0:.1f}s on {device}\n")

    # Evaluate
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_dl:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds.extend(out.logits.argmax(-1).cpu().numpy())
    y_pred = [id2label[p] for p in preds]
    y_true = test_df["label"].tolist()

    macro = f1_score(y_true, y_pred, average="macro")
    print("=== Held-out evaluation: fine-tuned MiniLM ===")
    print(f"Macro-F1: {macro:.3f}\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    print(f"TF-IDF baseline:          {TFIDF_BASELINE:.3f}")
    print(f"Frozen embeddings + MLP:  {FROZEN_EMBED_MLP:.3f}")
    print(f"Fine-tuned MiniLM:        {macro:.3f}  "
          f"({'up' if macro >= TFIDF_BASELINE else 'down'} "
          f"{abs(macro - TFIDF_BASELINE):.3f} vs TF-IDF)")

    out_dir = MODELS_DIR / "finetuned_minilm"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved fine-tuned model -> {out_dir}")


if __name__ == "__main__":
    main()
