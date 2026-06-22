"""
Zero-shot baseline: prompt Groq's llama-3.3-70b-versatile to classify each
test example with no task-specific training. Uses the SAME locked test set as
the fine-tuned model (data/test_split.csv) so the comparison is fair.

Requires GROQ_API_KEY in the environment.
Merges results into evaluation_results.json under "baseline".
"""
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["analysis", "hot_take", "reaction"]
MODEL = "llama-3.3-70b-versatile"

PROMPT = """You are classifying r/soccer comments into exactly one label.

analysis = a structured argument backed by specific, verifiable evidence (stats, formations, named match events, history). The evidence is load-bearing.
hot_take = a bold, confident opinion with no evidence or only decorative/cherry-picked evidence. It asserts rather than argues.
reaction = an immediate emotional response to an event. Little to no argument, just a feeling in the moment.

Reply with ONLY one word: analysis, hot_take, or reaction. No punctuation, no explanation.

Comment: {text}"""

def parse(resp):
    r = resp.strip().lower()
    r = re.sub(r"[^a-z_ ]", "", r)
    for lab in LABELS:
        if lab in r or lab.replace("_", " ") in r:
            return lab
    return None

def main():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set. Run: export GROQ_API_KEY=...")
    client = Groq(api_key=key)

    test = pd.read_csv(ROOT / "data" / "test_split.csv")
    preds, unparsed = [], 0
    for i, row in enumerate(test.itertuples(), 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(text=row.text)}],
            temperature=0, max_tokens=5,
        )
        raw = resp.choices[0].message.content
        lab = parse(raw)
        if lab is None:
            unparsed += 1
            lab = "hot_take"  # fallback so the row still scores
        preds.append(lab)
        print(f"  [{i}/{len(test)}] true={row.label:9s} pred={lab}")
        time.sleep(0.3)  # stay under free-tier rate limit

    y_true = test["label"].tolist()
    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, labels=LABELS,
                                   target_names=LABELS, output_dict=True, zero_division=0)
    print(f"\nbaseline accuracy: {acc:.3f}  (unparsed: {unparsed}/{len(test)})")
    print(classification_report(y_true, preds, labels=LABELS, target_names=LABELS, zero_division=0))

    rp = ROOT / "evaluation_results.json"
    results = json.loads(rp.read_text()) if rp.exists() else {}
    results["baseline"] = {
        "model": MODEL,
        "approach": "zero-shot prompt, temperature 0",
        "accuracy": round(float(acc), 4),
        "unparsed": unparsed,
        "per_class": {l: {k: round(report[l][k], 4) for k in ("precision", "recall", "f1-score")}
                      for l in LABELS},
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
    }
    rp.write_text(json.dumps(results, indent=2))
    print("wrote evaluation_results.json (baseline section)")

if __name__ == "__main__":
    main()
