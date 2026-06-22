# TakeMeter — Soccer Discourse Classifier

Fine-tuned text classifier that rates the quality of "takes" in r/soccer, sorting posts into `analysis`, `hot_take`, or `reaction`.

## Demo video

- [Demo part 1](https://www.loom.com/share/889fff3888ec456ab15960f960089001)
- [Demo part 2](https://www.loom.com/share/53a9a658bbf348eb9bbdf7d69b1eab4b)

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

- **Source:** r/soccer (plus a handful of comments surfaced via r/Barca, r/psg crossposts). Comments were collected manually by copy-pasting from public threads — a mix of post-match threads, stats/record threads, club-crisis threads, and match-clip threads. Teams/topics span Barcelona, Real Madrid, Man Utd, Man City, Chelsea, Arsenal, Spurs, PSG, Bayern, Cape Verde/Uruguay — across La Liga, Premier League, Ligue 1, Champions League and the World Cup, to avoid the model memorizing a single club's vocabulary.
- **Process:** Each comment was read individually and labeled by hand against the definitions in [planning.md](planning.md). Jokes, one-word replies, non-English comments, and bot/ad posts were skipped. Borderline cases were noted in the CSV `notes` column. Thread *type* was deliberately matched to label need — stats/crisis threads for `analysis`, clip/drama threads for `reaction` — to keep the classes balanced.
- **Label distribution (195 examples):**

  | Label | Count | Share |
  |-------|-------|-------|
  | analysis | 65 | 33% |
  | hot_take | 65 | 33% |
  | reaction | 65 | 33% |

  Perfectly balanced — no label exceeds the 70% imbalance threshold.
- **3 difficult-to-label examples:** documented with decisions in [planning.md](planning.md) (Hard Edge Cases). Summary: a values-based argument with no verifiable evidence (→ `hot_take`), a one-line claim backed by a real stat (→ `analysis`), and a fact-laden grievance screed (→ `hot_take`).

## Fine-tuning approach

- **Base model:** distilbert-base-uncased (HuggingFace) — a small, fast, general English language model with no idea what these labels mean until we train it.
- **Training setup:** fine-tuned with the HuggingFace `Trainer` on the 136-example training split. Batch size 16, learning rate 5e-5, 10 epochs, 10% warmup. Ran locally on an Apple-Silicon GPU (MPS) in ~8 minutes; training loss fell from 1.09 to 0.02.
- **Key hyperparameter decision — epochs (3 → 10) and learning rate (2e-5 → 5e-5):** The notebook default of 3 epochs at lr 2e-5 *underfit* badly. With only 136 training examples and batch size 16, 3 epochs is just 27 optimizer steps — the loss stayed near random (~1.05 vs. the 1.10 you get from pure guessing) and the model never once predicted `reaction` (its F1 was 0.00, overall accuracy 43%). I raised epochs to 10 and the learning rate to 5e-5 so the freshly-initialized classifier head had enough steps to actually converge. After the change, loss dropped to 0.02 and accuracy rose to 60%. Trade-off: 10 epochs on this little data risks overfitting (and the results below suggest it does), but underfitting was the bigger, more visible failure to fix first.

## Baseline

- **Model:** Groq `llama-3.3-70b-versatile`, zero-shot (no training, temperature 0).
- **Prompt used:** the model is given the three label definitions verbatim from this README and instructed to reply with exactly one label word:

  ```
  You are classifying r/soccer comments into exactly one label.

  analysis = a structured argument backed by specific, verifiable evidence (stats, formations, named match events, history). The evidence is load-bearing.
  hot_take = a bold, confident opinion with no evidence or only decorative/cherry-picked evidence. It asserts rather than argues.
  reaction = an immediate emotional response to an event. Little to no argument, just a feeling.

  Reply with ONLY one word: analysis, hot_take, or reaction.

  Comment: {text}
  ```
- **How results collected:** each of the 30 test comments was sent to Groq one at a time; the one-word reply was parsed to a label. 0 of 30 replies were unparseable. The baseline runs on the **exact same test split** as the fine-tuned model, so the comparison is fair.

## Evaluation report

Test set: 30 comments (10 analysis / 10 hot_take / 10 reaction). Full numbers in [evaluation_results.json](evaluation_results.json).

### Overall accuracy

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Groq baseline (zero-shot llama-3.3-70b) | **66.7%** | 0.66 |
| Fine-tuned DistilBERT | 60.0% | 0.58 |

Random guessing on 3 balanced classes = 33%. Both models clear that comfortably. **The zero-shot baseline beat the fine-tuned model by 6.7 points** — see Reflection for why.

### Per-class metrics

| Label | Groq P / R / F1 | Fine-tuned P / R / F1 |
|-------|------------------|------------------------|
| analysis | 0.88 / 0.70 / **0.78** | 0.58 / 0.70 / 0.64 |
| hot_take | 0.80 / 0.40 / **0.53** | 0.50 / 0.30 / 0.38 |
| reaction | 0.53 / 0.90 / 0.67 | 0.67 / 0.80 / **0.73** |

Both models find **`hot_take` the hardest** (lowest F1 for each). The fine-tuned model's one win is `reaction` (0.73 vs 0.67).

### Confusion matrix (fine-tuned)

Rows = true label, columns = what the model predicted. Diagonal = correct.

| true ↓ / pred → | analysis | hot_take | reaction |
|-----------------|----------|----------|----------|
| **analysis**    | 7        | 1        | 2        |
| **hot_take**    | 4        | 3        | 3        |
| **reaction**    | 1        | 1        | 8        |

(Also committed as [confusion_matrix.png](confusion_matrix.png).)

**The story this tells:** `hot_take` is the problem row — only **3 of 10** correct. It bleeds in *both* directions: 4 hot_takes were called `analysis`, 3 were called `reaction`. That is the central finding — `hot_take` sits between the other two classes and the model can't hold the boundary. `analysis` (7/10) and `reaction` (8/10) are learned reasonably well.

### 3 wrong predictions analyzed

1. **"Before the match everyone said Arsenal will get battered... 1-1 after 120 minutes PSG don't score from anything that isn't a penalty. After the match: Arsenal are rubbish why didn't they attack. I'm not sure if people are biased or stupid."**
   - True: `hot_take` · Predicted: `analysis` · confidence **0.97**
   - **Why it failed:** the comment is long, references specific match events (score, the penalty), and is structured like an argument — all surface features of `analysis`. But it's really a sarcastic opinion with no genuine reasoning. The model learned "mentions match details + long = analysis" and got fooled. This is the `hot_take → analysis` confusion, the matrix's biggest leak.

2. **"That was the old Chelsea. We're just shit now."**
   - True: `hot_take` · Predicted: `reaction` · confidence **0.96**
   - **Why it failed:** short, emotional, profane — all surface features of `reaction`. But it's actually a standing judgment about the team's quality (a hot take), not an in-the-moment feeling about an event. The model read the *tone*, not the *content*. This is the `hot_take → reaction` confusion.

3. **"Helps to play 12 fewer matches from Aug to Feb. 29 hits a lot different on a squad than 41."**
   - True: `analysis` · Predicted: `reaction` · confidence **0.82**
   - **Why it failed:** it's a genuine evidence-based point (fixture congestion, specific match counts), but it's *short and casually phrased*. The model associates short + casual with `reaction`, so the brevity overrode the actual argument. Shows the model leans on length as a proxy for `analysis`.

**Is this a labeling problem or a data problem?** I labeled these consistently with my own definitions — the model still missed them. So the issue is the *training data distribution and the boundary itself*: `hot_take` is genuinely the hardest class to pin down (it overlaps `analysis` when it cites facts and `reaction` when it's emotional), and 45 training examples per class isn't enough for DistilBERT to learn that it should weigh *reasoning quality* over *surface features* like length and tone.

### Sample classifications

5 test comments run through the fine-tuned model with its confidence:

| Comment (trimmed) | Predicted | Confidence | Correct? |
|-------------------|-----------|-----------|----------|
| "Zubimendi looked exhausted, he should bench the next few games. He's been overplayed and it shows..." | analysis | 0.99 | ✅ |
| "Real Madrid tries to retain 50% of future transfer fees and they have a lot of value in players that way..." | analysis | 0.99 | ✅ |
| "11 points of the improvement have come from Carrick's six games in charge." | analysis | 0.98 | ✅ |
| "That was the old Chelsea. We're just shit now." | reaction | 0.96 | ❌ (was hot_take) |
| "Before the match everyone said Arsenal will get battered..." | analysis | 0.97 | ❌ (was hot_take) |

The first one is a reasonable, confident `analysis` call: the comment makes a specific claim about a player (Zubimendi is fatigued), backs it with an observation (overplayed, it shows), and recommends an action — exactly the structured-argument-with-evidence pattern the `analysis` label is meant to capture.

### Reflection: learned vs. intended

**What I intended:** the model should judge a comment by *reasoning quality* — does it argue from verifiable evidence (`analysis`), assert without it (`hot_take`), or just react emotionally (`reaction`)?

**What the model actually learned:** surface features that *correlate* with my labels in the training data, not the underlying concept.
- **Length and structure → analysis.** Long comments that name match events get called `analysis` even when they're sarcastic rants (wrong prediction #1).
- **Short and emotional → reaction.** Brief, punchy, or profane comments get called `reaction` even when they're standing opinions (wrong prediction #2) or actual evidence-based points (#3).
- **`hot_take` has no clean surface signal of its own**, so it loses to whichever neighbor a comment superficially resembles — exactly why it scored worst (F1 0.38) and bled both ways in the confusion matrix.

This is the gap the project is about: my labels encode *why* a take is good or bad, but with 136 examples the model latched onto *what* good and bad takes tend to look like. The notably high confidence on its wrong answers (0.96, 0.97) confirms it — it isn't unsure, it's confidently using the wrong rule.

## Spec reflection

- **One way the spec helped:** writing the label definitions and the explicit `analysis`-vs-`hot_take` decision rule in [planning.md](planning.md) *before* annotating kept my 195 labels consistent. When I hit borderline cases during collection (a one-stat opinion, a fact-laden rant), I had a written rule to apply instead of deciding ad hoc — which is exactly why the failure analysis above could conclude "this is a boundary problem, not an inconsistency problem."
- **One way the implementation diverged:** the plan assumed the notebook's default hyperparameters would work and that fine-tuning would beat the baseline. Neither held — 3 epochs underfit (forcing the 10-epoch / 5e-5 change), and the zero-shot baseline ended up *winning*. The plan treated the baseline as a formality to beat; in reality it became the most informative result, showing that a 70B zero-shot model generalizes better than DistilBERT fine-tuned on only ~45 examples per class.

## AI usage

- **Label stress-testing & edge-case design:** I gave an AI assistant my draft label definitions and asked it to surface borderline soccer comments and help write the `analysis`-vs-`hot_take` decision rule. It proposed the "load-bearing vs. decorative evidence" framing; I adopted it but tightened the wording and chose the final example posts myself.
- **Annotation assistance (disclosed):** during collection I pasted batches of raw r/soccer comments to an AI and had it suggest a label for each. I reviewed and corrected every suggestion by hand against my definitions before it entered the CSV — several were re-labeled (e.g. reasoned-but-evidence-free opinions it called `analysis`, which I moved to `hot_take`). No label entered the dataset unreviewed.
- **Failure-pattern analysis:** I gave the AI my 12 misclassified test examples and asked for systematic patterns. It flagged the "length/structure → analysis, short/emotional → reaction" tendency; I verified each pattern by re-reading the comments and the confusion matrix before writing the Reflection.

## Repo contents

- `planning.md` — design doc (labels, edge cases, metrics, AI plan)
- `data/takemeter_dataset.csv` — the 195 labeled examples
- `data/test_split.csv`, `data/test_predictions.csv` — locked test set + per-example predictions
- `takemeter_local.ipynb` — the notebook I actually ran (commented, runs locally in VS Code)
- `scripts/train_eval.py`, `scripts/baseline_groq.py` — script versions of the pipeline
- `evaluation_results.json`, `confusion_matrix.png` — evaluation outputs
