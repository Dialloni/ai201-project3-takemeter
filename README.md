# TakeMeter — Soccer Discourse Classifier

Fine-tuned text classifier that rates the quality of "takes" in r/soccer, sorting posts into `analysis`, `hot_take`, or `reaction`.

> Status: in progress. Sections marked TODO get filled as milestones complete.

## Community

r/soccer (Reddit). Chosen because the discourse ranges from detailed tactical/statistical breakdowns to one-line emotional outbursts to confident no-evidence opinions — a spread the community itself argues over ("real analysis vs. just a hot take"). See [planning.md](planning.md) for full reasoning.

## Label taxonomy

| Label | Definition | Example |
|-------|-----------|---------|
| `analysis` | Structured argument backed by specific, load-bearing evidence (stats, formations, match events, history). | "City's xG was 2.8 vs Arsenal's 0.6 — they dominated despite the scoreline. Rodri shut down the counters." |
| `hot_take` | Bold confident opinion with no/decorative evidence. Asserts, doesn't argue. | "Haaland is overrated, he disappears in big games." |
| `reaction` | Immediate emotional response to an event. Feeling in the moment, little argument. | "WHAT A GOAL. I'm shaking. Bellingham is HIM." |

Second example per label and edge-case rules: see [planning.md](planning.md).

## Data collection

- **Source:** TODO — r/soccer threads collected (which threads, date range)
- **Process:** TODO — manual copy-paste, label per planning.md definitions
- **Label distribution:** TODO — count per label
- **3 difficult-to-label examples:** TODO — see planning.md Hard Edge Cases

## Fine-tuning approach

- **Base model:** distilbert-base-uncased (HuggingFace)
- **Training setup:** TODO — epochs, learning rate, batch size
- **Key hyperparameter decision:** TODO

## Baseline

- **Model:** Groq llama-3.3-70b-versatile, zero-shot
- **Prompt used:** TODO
- **How results collected:** TODO

## Evaluation report

### Overall accuracy

| Model | Accuracy |
|-------|----------|
| Groq baseline | TODO |
| Fine-tuned DistilBERT | TODO |

### Per-class metrics

TODO — precision / recall / F1 per label, both models.

### Confusion matrix (fine-tuned)

TODO — markdown table, rows = true, cols = predicted. (Also committed as confusion_matrix.png.)

### 3 wrong predictions analyzed

TODO — for each: the post, true label, predicted label, why it failed.

### Sample classifications

TODO — 3–5 posts with predicted label + confidence; explain one correct one.

### Reflection: learned vs. intended

TODO — gap between label definitions and what the model's boundary actually captured.

## Spec reflection

TODO — one way the spec helped, one way implementation diverged and why.

## AI usage

TODO — ≥2 specific instances: what I directed the AI to do, what it produced, what I changed/overrode. Disclose any annotation pre-labeling.

## Repo contents

- `planning.md` — design doc
- `data/takemeter_dataset.csv` — labeled dataset
- `Copy_of_ai201_project3_takemeter_starter_clean.ipynb` — Colab training notebook
- `evaluation_results.json`, `confusion_matrix.png` — outputs (added after Milestone 5)
