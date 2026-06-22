"""
Fine-tune distilbert-base-uncased on the TakeMeter dataset and evaluate on a
held-out test set. Mirrors the Colab notebook: 70/15/15 stratified split,
3 epochs, lr 2e-5, batch size 16.

Outputs (written to repo root):
  - confusion_matrix.png        confusion matrix for the fine-tuned model
  - evaluation_results.json     fine-tuned metrics (baseline merged in later)
  - data/test_split.csv         the locked test set (shared with the baseline)
  - data/test_predictions.csv   per-example predictions + confidence for analysis
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LABEL_MAP = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABELS = list(LABEL_MAP.keys())
MODEL = "distilbert-base-uncased"
SEED = 42

# ── 1. load + stratified 70/15/15 split ──────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "takemeter_dataset.csv")
df = df[["text", "label"]].dropna()
df["y"] = df["label"].map(LABEL_MAP)
assert df["y"].notna().all(), "a label in the CSV is not in LABEL_MAP"

train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["y"], random_state=SEED)
print(f"split  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
test_df.to_csv(ROOT / "data" / "test_split.csv", index=False)

# ── 2. tokenize ──────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL)

def encode(texts):
    return tok(list(texts), truncation=True, padding=True, max_length=256)

class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

train_ds = DS(encode(train_df["text"]), train_df["y"])
test_ds = DS(encode(test_df["text"]), test_df["y"])

# ── 3. fine-tune ─────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL_MAP)

# Hyperparameter decision: the notebook defaults (3 epochs, lr 2e-5) underfit
# badly on this small 136-example train set — only 27 optimizer steps, train
# loss stayed near random (~1.05 vs ln(3)=1.10) and `reaction` was never
# predicted. Raised to 10 epochs + lr 5e-5 to give the classifier head enough
# steps to converge. See README hyperparameter section.
args = TrainingArguments(
    output_dir=str(ROOT / "scripts" / "_trainer_out"),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds)
trainer.train()

# ── 4. evaluate on test ──────────────────────────────────────────────────────
pred = trainer.predict(test_ds)
probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
y_pred = probs.argmax(1)
y_true = test_df["y"].to_numpy()
conf = probs.max(1)

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=LABELS,
                               output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred)
print(f"\nfine-tuned accuracy: {acc:.3f}")
print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

# per-example dump for error analysis + sample classifications
out = test_df.copy()
out["pred"] = [ID2LABEL[i] for i in y_pred]
out["confidence"] = conf.round(4)
out["correct"] = out["y"].to_numpy() == y_pred
out[["text", "label", "pred", "confidence", "correct"]].to_csv(
    ROOT / "data" / "test_predictions.csv", index=False)

# ── 5. confusion matrix png ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Fine-tuned DistilBERT — Confusion Matrix")
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im); fig.tight_layout()
fig.savefig(ROOT / "confusion_matrix.png", dpi=150)
print("saved confusion_matrix.png")

# ── 6. write/merge evaluation_results.json ───────────────────────────────────
results_path = ROOT / "evaluation_results.json"
results = json.loads(results_path.read_text()) if results_path.exists() else {}
results["test_set_size"] = int(len(test_df))
results["labels"] = LABELS
results["fine_tuned"] = {
    "model": MODEL,
    "accuracy": round(float(acc), 4),
    "per_class": {l: {k: round(report[l][k], 4) for k in ("precision", "recall", "f1-score")}
                  for l in LABELS},
    "macro_f1": round(report["macro avg"]["f1-score"], 4),
    "confusion_matrix": {"rows_true_cols_pred": LABELS, "matrix": cm.tolist()},
}
results_path.write_text(json.dumps(results, indent=2))
print("wrote evaluation_results.json (fine_tuned section)")
