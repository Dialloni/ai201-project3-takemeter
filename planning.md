# TakeMeter — Planning

> Design doc. Written before data collection. Update before any stretch feature.

## 1. Community

**Choice:** r/soccer (Reddit).

**Why:** r/soccer is high-volume, text-heavy, and the discourse quality varies enormously — from detailed tactical breakdowns to one-line emotional outbursts to confident opinions with zero backing. The community itself constantly argues over "actual analysis vs. just a hot take," so the distinction I'm measuring is one the community already recognizes and cares about. Match threads, post-match threads, and transfer rumor threads give an easy mix of all three label types.

## 2. Labels (3)

Mutually exclusive. None expected above ~50% of the dataset.

### `analysis`
A post making a structured argument backed by **specific, load-bearing evidence** — stats (xG, possession, pass %), formations, named match events, or historical comparison. Removing the evidence would collapse the argument.

- Example A: "City's xG was 2.8 vs Arsenal's 0.6 — they dominated despite the 1-1 scoreline. Rodri controlling the half-spaces shut down Arsenal's counters all game."
- Example B: "Saka's been drifting inside more this season; that's why Arsenal's right side looks narrow and White has to overlap. It worked vs low blocks but leaves them exposed on the break."

### `hot_take`
A bold, confident opinion stated with **no evidence or only decorative evidence**. The post asserts rather than argues. The claim might be true, but it isn't reasoned.

- Example A: "Haaland is overrated, he disappears in big games. Mbappé is on another level and it's not close."
- Example B: "Ten Hag is clueless, United have lost 4 of their last 6." (one stat, but cherry-picked to back a rant, not to reason → still hot_take)

### `reaction`
An immediate **emotional response** to a specific event. Little to no argument — the post is expressing a feeling in the moment.

- Example A: "WHAT A GOAL. I'm actually shaking. Bellingham is HIM."
- Example B: "no no no no not again, I can't watch this team anymore 😭"

## 3. Hard edge cases

**Boundary that will be ambiguous:** `analysis` vs `hot_take` — posts that cite ONE statistic.

**Decision rule:** Evidence must be *specific and load-bearing* — the argument depends on it — to count as `analysis`. A vague or cherry-picked stat that just decorates an opinion (sounds credible but isn't doing reasoning work) → `hot_take`.

**Second boundary:** `reaction` vs `hot_take` — an emotional post that also contains an opinion ("Haaland trash 😭"). Rule: if the dominant content is a *feeling in the moment* tied to an event → `reaction`; if it's a *standing opinion/judgment* about a player/team → `hot_take`.

**3 real difficult examples encountered during annotation:**

1. **"Yes the World Cup is once in a lifetime... but becoming a father and seeing your child's birth goes far beyond that... he'd trade 10000 World Cup trophies to be there."** (Pierron thread)
   - Candidate labels: `analysis` vs `hot_take`. It is clearly *reasoned* — it builds a case rather than just asserting.
   - **Decision → `hot_take`.** The reasoning is values-based, not *verifiable evidence* (no stats, no match events). My rule: `analysis` requires specific, checkable evidence. A well-argued opinion with no verifiable facts is still a hot take. This was the single hardest recurring boundary.

2. **"Lewandowski is a proper scorer. Top 10 at both Bayern and Barca."** (Lewa stats thread)
   - Candidate labels: `reaction` vs `analysis`. Very short, almost a one-liner — feels like a `reaction`.
   - **Decision → `analysis`.** Length isn't the criterion; *evidence is*. It makes a claim ("proper scorer") and backs it with a specific, verifiable fact (top 10 at both clubs). Short but load-bearing evidence = analysis.

3. **"QSG's every victory being linked to sportswashing is fair... Ligue 1 killing their domestic competition, 0 ffp regulations, spending 200mil every summer for over a decade..."** (CL final thread)
   - Candidate labels: `analysis` vs `hot_take`. It cites specifics (FFP, spending figures) like analysis, but reads as an angry screed.
   - **Decision → `hot_take`.** The "evidence" is decorative/rhetorical — selected to fuel a grievance rather than to reason toward a conclusion. Per my edge-case rule (decorative vs load-bearing evidence), this is a hot take dressed in facts.

## 4. Data collection plan

- **Source:** r/soccer — match threads, post-match threads, transfer rumor threads, daily discussion. Public posts/comments only.
- **Method:** Manual copy-paste into `data/takemeter_dataset.csv` (columns: `text`, `label`, `notes`). Keeps me close to the data.
- **Target:** ~200 examples, aiming ~60–70 per label so no label exceeds 70%.
- **If a label is underrepresented after 200:** go back to threads likely to contain it (tactical threads → more `analysis`; live match threads → more `reaction`) and collect more until each label ≥ 20%.

## 5. Evaluation metrics

- **Accuracy** (both models) — quick overall signal, but not enough alone because classes may be imbalanced and accuracy hides per-class failure.
- **Per-class precision / recall / F1** — tells me *which* distinction the model learned. I care most about `analysis` recall (catching real analysis) and `hot_take` precision (not over-flagging everything as a hot take).
- **Confusion matrix** — to see the *direction* of errors, especially the expected `analysis ↔ hot_take` confusion.

Why these: the whole point is whether the model learned the *boundary*, not just the majority class. Per-class F1 + confusion matrix expose that; accuracy alone would not.

## 6. Definition of success

- **Good enough for deployment:** fine-tuned model beats the Groq zero-shot baseline on overall accuracy AND every class has F1 ≥ 0.65.
- **Genuinely useful:** `analysis` F1 ≥ 0.70 specifically — that's the label a real "good take" highlighter tool would surface, so it's the one that has to work.
- Anything below baseline = investigate label leakage / imbalance / training bug before writing up.

## AI Tool Plan

- **Label stress-testing:** Give Claude my 3 label definitions + edge-case rules and ask for 5–10 boundary posts. If I can't classify them cleanly, tighten definitions BEFORE annotating 200.
- **Annotation assistance:** *Decision:* may use Groq/Claude to pre-label a batch, then review and correct every single one by hand. Will mark pre-labeled rows in `notes` and disclose in README. (If it slows me down or adds noise, skip and label fully manually.)
- **Failure analysis:** After fine-tuning, paste all wrong predictions into Claude and ask it to find systematic patterns (label pair confused, sarcasm, short posts). Verify each pattern myself by re-reading before putting it in the report.
