# E5: quality-constrained adaptive selective emotion-neuron steering

Status (2026-11-24): audit done; E5 implemented and tested (CPU tests, Stage 1 dry run); **no E5 GPU run yet**.
Every number below comes from already-used development data, never from E5 held-out data.

Contents: [1 Audit](#1-audit-of-the-existing-system) · [2 Design](#2-e5-design) · [3 Implementation](#3-implementation-map) ·
[4 Data](#4-data-and-leakage-prevention) · [5 Calibration](#5-calibration-procedure-fixed-before-any-calibration-data) ·
[6 Evaluation](#6-held-out-evaluation-and-statistics) · [7 Safety](#7-safety) · [8 Runbook](#8-runbook) ·
[9 Limitations](#9-limitations)

---

## 1. Audit of the existing system

All figures: `python3 -m generation_analysis.e5_audit` → `results/e5_selective/audit/`. Development data = the 527
messages of the exploratory (`results/steering_modes`, `results/steering_kg`, 226) and confirmatory
(`results/confirmatory`, 301) studies. Each message has E3 and E4 replies rated by both judges (Qwen2.5-7B and
Gemma-3-27B).

### 1.1 How E4 adapts strength

`EmotionSteeringService._steer_from_state` ([backend/emotion_service.py:374](../backend/emotion_service.py#L374))
calls `EmotionReasonerAgent.state_strength` ([agent.py:244](../chatbot_feature/emotion_reasoner/agent.py#L244)).
The steps are:

1. **Raw strength.** The strategy's base strength (0.4 gentle exploration up to 0.8 comfort/celebration; a crisis
   is steered toward love at 0.9) is multiplied by `0.8 + 0.4 × intensity`.
2. **Panel sensitivity.** Joy is multiplied by 1.35 and admiration by 0.8.
3. **History.** Persistent distress adds 0.15; a lifting mood subtracts 0.10.
4. **Cap.** The result is capped at 1.2, or 0.7 for Hinglish.
5. **State level.** Dividing by 1.2 gives a level between 0 and 1.
6. **Model range.** The level is mapped onto `STATE_ALPHA_RANGE`: 0.25–0.75 for the Llama family, 2–3 for
   Sarvam's tone panels.

Observed applied strength: median 0.54 (P10 0.43, P90 0.69) for the Llama family, and 2.58 for Sarvam.
E4 sends the steered reply unless `_check(always=True)` finds it broken. Broken means one of: not conversational, a
repeated-bigram rate more than 0.05 above the unsteered reply's, shorter than 0.4× the unsteered reply, or switched
from Hinglish to English.

### 1.2 Why E5-old (`policy` mode) almost never steers

Decision categories for the 226 messages of `results/steering_modes`, policy mode (`audit/policy_coverage.csv`):

| model | calibrated cell = 0 | cell too small | crisis | no panel | gate skipped / rejected | **steered** |
|---|---|---|---|---|---|---|
| Llama 3.1 | 69% | 6% | 7% | – | 1.3% | **16%** |
| EmoLLaMA | 79% | 6% | 7% | – | 0% | **8%** |
| Llama-2 | 78% | 6% | 7% | – | 0.4% | **8%** |
| Sarvam | 32% | – | 7% | 61% | – | **0%** |

Coverage is decided **before generation** by calibration cells set to α 0. The run-time gate removes almost nothing.
The ceiling rule (skip when the unsteered reply's classifier score is ≥ 0.30) fired twice in 226.

The cells were zeroed by `generation_analysis/recalibrate_policy.py` (policy v2). The causes:

- **Small cells.** 19 messages per strategy cell (5 for joy/Hinglish), judged by one judge.
- **Strict rules on small cells.** A per-metric tolerance of −0.05 on empathy, relevance and fluency. A pairwise
  preference of at least 0.45: steering raised tone, but judges' holistic preference was often 0.40–0.44. A safety
  margin requiring the next stronger strength to pass too. Example: admiration/English at α 0.25 gained +0.13 tone
  with quality unchanged and was still rejected at preference 0.42
  (`results/llama31_8b_generation_emotion/causal_steering/recalibration/sweep_by_cell.csv`).
- **Hinglish.** Every Hinglish cell failed: 44–78% of replies harmed at α ≤ 0.5.
- **Sarvam.** The policy is `steering_enabled: false`, and its tone-panel names do not match the planned response
  emotion. That is the "no panel" column.

So low coverage comes from strategy and language restrictions plus small-sample calibration. It is not caused by
confidence thresholds, alpha values, run-time rejection or weak probes.

### 1.3 `steered_share`: attempted vs retained

`steered_share = mean(applied_alpha > 0)` (`steering_modes.summarise`, `confirmatory.summarise`). `applied_alpha` is
set only when the candidate passed the run-time checks and was sent, so **`steered_share` is the retained rate**.
Attempts are `planned_alpha > 0`.

| model | E4 attempted | E4 retained | E4 rejected (of attempts) |
|---|---|---|---|
| Llama 3.1 | 100% | 87–89% | 11–13% (mostly repetition) |
| EmoLLaMA | 100% | 96% | 4% |
| Llama-2 | 100% | 96% | 4% |
| Sarvam | 100% | 92–95% | 5–8% |

### 1.4 The fallback mechanism and what it can detect

`_check` ([backend/emotion_service.py:431](../backend/emotion_service.py#L431)) has two paths. The `always` path uses
only the "broken reply" rules. The policy path adds the ceiling, optional length and language rules, and optionally a
classifier gain of at least 0.05. The `policy` gate in use is `ceiling_and_safeguards`.

**Key finding (`audit/acceptance_predictors.csv`).** Among E4's retained replies, no cheap candidate feature predicts
a judged harm better than chance. Judged harm means any quality or safety rating falling more than 0.5. Features
tested: word ratio, length ratio, repetition increase, language change, classifier gain, unsteered classifier score,
alpha, intensity. AUC was 0.45–0.58 on all four models.

Risky-phrasing markers (familiarity, dependency, diagnosis, false reassurance) appear in ≤ 4% of replies
(`audit/markers.csv`).

**Implication:** a lightweight acceptance step cannot catch the replies steering damages. E5's quality control has to
come from *where* it steers (context selection), not from post-hoc filtering. A judge-based response critic (E6) is
the only lever left for per-reply detection.

### 1.5 Where E4 helps and harms (development data, E4 − E3, both judges)

Pooled per model (`audit/audit.json`):

| model | tone | quality (5-metric mean) | safety | harmed (of steered) | preference E4 over E3 |
|---|---|---|---|---|---|
| Llama 3.1 | +0.39 [+0.34, +0.44] | −0.05 [−0.08, −0.02] | −0.11 [−0.15, −0.07] | 20% | 0.50 |
| EmoLLaMA | +0.34 [+0.28, +0.41] | +0.13 [+0.06, +0.20] | +0.11 [+0.03, +0.19] | 20% | 0.52 |
| Llama-2 | +0.31 [+0.27, +0.35] | −0.23 [−0.27, −0.19] | −0.31 [−0.36, −0.26] | 37% | 0.32 |
| Sarvam | +0.13 [+0.09, +0.17] | −0.02 [−0.05, +0.01] | −0.07 [−0.10, −0.04] | 18% | 0.44 |

Per strategy × language (`audit/cell_effects.csv`) the effects are clearly structured:

- **Llama 3.1, English, love-target strategies** (comfort, grounding, gentle exploration, validation): about +0.3
  tone, quality about 0, safety −0.02 to −0.09, preference 0.55–0.64.
- **Llama 3.1, celebratory (joy):** +0.81 tone but safety −0.13 and preference 0.32.
- **Llama 3.1, Hinglish:** safety −0.22 (vs −0.07 in English), with 18–43% harmed. Typical failures are semantic
  drift ("aapki dukaan…", i.e. "your shop") and cheerfulness about guilt.
- **Llama-2:** loses quality in every cell. Celebratory English is −0.82 quality with 75% harmed. Even α ≤ 0.45
  harms 26% of steered replies (`audit/model_specific.json`, dose-response).
- **EmoLLaMA:** the KG prompt causes degenerate replies (numbers or fragments): 7–9% with the KG prompt vs 3.3–3.5%
  with KG + steering. Gentle exploration, grounding and reflective validation lose 0.2–0.75 quality with the KG
  prompt vs the plain prompt. Steering repairs part of this, so E4 > E3 on quality for EmoLLaMA.
- **Sarvam:** small tone gains (English +0.11, Hinglish +0.17) and safety −0.07 in both languages.

### 1.6 Is the classifier a usable "emotional deficit" signal?

User-message detection was already validated (`results/emotion_trajectory/validation.json`): English accuracy 0.80,
synthetic Hinglish 0.57, ECE 0.15 / 0.06. **On assistant replies** (`audit/classifier_on_replies.csv`), the gate
classifier's target score correlated with the judges' tone as follows:

| | English | Hinglish |
|---|---|---|
| Spearman with judges' tone | 0.52–0.69 | −0.02 to 0.55 (Sarvam −0.02) |
| Inter-judge Spearman (for reference) | 0.65–0.76 | 0.51–0.78 |

So the deficit signal is usable only in English. As a *moderator* (`audit/deficit_moderation.csv`): when the
unsteered reply already scores ≥ 0.30, steering is preferred less often (Llama 0.39 vs 0.56; Llama-2 0.21 vs 0.37)
and, for Llama-2, harms more (51% vs ~30%). Tone gains do **not** grow with the deficit. This is consistent with the
old moderator table (`recalibration/moderators.csv`).

The deficit gate is therefore a quality protector, English-only. A low score alone never triggers steering.

### 1.7 Causal-validation evidence (reused, not rebuilt)

`generation_analysis/neuron_validation.py` (`results/<tag>_generation_emotion/neuron_validation/report.json`) shows
panel-level causality on all four models:

- **Llama:** panel at α 1.5 gives +1.44 vs −0.04 for matched random neurons; reverse steering gives −0.78; clamping
  gives −0.27; sham gives 0.
- **EmoLLaMA:** weaker.
- **Sarvam:** causal at panel level only.

In Sarvam's tests, subsets of the top-ranked neurons were no better than random subsets. Single-neuron screens exist
(`neuron_validation/single/`), but cover only listed neurons at 32× strength. **There is no causal evidence that
choosing neurons or layers per turn would help, so E5 does not add dynamic neuron or layer selection.** The v3
late-layer band (L22–31) was quality-safe but weaker (`results/steering_v3/decisions.json`).

### 1.8 A measurement caveat for every E4 − E3 contrast

E4 decodes the unsteered and steered replies in one batch, while E3 decodes alone. Greedy batched decoding differs
numerically from single-row decoding. E4's batched unsteered row equals the E3 reply for only 60–71% of Llama-family
messages and **7–10% of Sarvam messages** (`audit/batch_consistency.csv`). The prompts are identical, so this is
arithmetic, not prompting.

Part of every past E4 − E3 difference is therefore batch noise. E5 judges the batched unsteered rows as controls
(`E4_batch0`, `E5_batch0`) to measure that noise floor.

### 1.9 Reused vs new

| Reused unchanged | New for E5 |
|---|---|
| Neuron panels (`results/<tag>_generation_emotion/steering_panels/`), steering hooks and batched decoding (`runtime.generate_candidates`) | Risk screen for steering eligibility (`risk_screen.py`) |
| E4's state-scaled strength (`state_strength`, `STATE_ALPHA_RANGE`) as the base of every multiplier | Per-model, per-context policy with strength multipliers (`selective.py`) |
| Gate classifier `ReplyEmotionScorer` (deficit signal) | English-only deficit gate; acceptance = E4's checks + language/script, length cap, added risky phrasing |
| E1–E5-old paths (`begin_turn`/`respond`; all modes and prompt variants) | `SelectiveSteeringService.run_selective` (subclass; existing paths untouched) |
| Judges, rubrics, statistics (`steering_lab.Judge`, `v3_common`, `paired_summary`, `holm`), lock/resume, registration pattern | Validation and strategy-adherence rubrics, Hindi naturalness, judge answer cache, calibration procedure (`e5_fit.py`) |

No existing file was modified. E1–E5-old behave exactly as before, which the existing 191 tests confirm.

---

## 2. E5 design

Per turn, four steps run in order, each with a logged reason
(`chatbot_feature/emotion_reasoner/selective.py`, `backend/selective_service.py`):

1. **Safety (pre-generation, hard).** Steering is disabled if the Emotion Reasoner flags a crisis, or if the risk
   screen returns any category except `none` or `figurative`. The steering-disabling categories are disclosure,
   harm to others, abuse or danger, medical emergency, indirect risk and mention. The reply is then the KG-only
   reply (E3's code path).
2. **Context (pre-generation).** The frozen model-specific cell for (strategy, language) applies; otherwise the
   language's residual cell. The cell gives a strength multiplier in {1.0, 0.75, 0.5} of E4's state-scaled α, or
   *no steering*. **No development evidence means no steering.**
3. **Deficit (post-generation, English only, if calibration chose it).** The unsteered and steered replies are
   decoded in one batch, as in E4. If the unsteered reply already expresses the target (classifier score ≥ the
   frozen threshold), the unsteered reply is sent.
4. **Acceptance (post-generation).** The candidate must pass E4's checks plus: same language and script as the
   unsteered reply, an optional word-ratio cap, and no risky phrasing that the unsteered reply lacked. If it fails,
   the batched unsteered reply is sent.

**Panel choice.** E5 uses the KG-selected target's panel (Sarvam: the strategy's tone panel). It does not swap
panels: joy and admiration are allowed only where calibration admits their cells.

**Strength.** E5 keeps E4's adaptive mechanism and scales it per context.

**Why this addresses both earlier failures:**

- **vs E4:** contexts where steering costs quality or safety are excluded by data. Llama-2 and Llama Hinglish are the
  examples.
- **vs E5-old:** the cells are estimated with shrinkage on 527+ messages and two judges, instead of one judge on 19
  messages. The quality constraints are the user's stated list (appropriateness, empathy, relevance, fluency,
  naturalness, validation, and safety separately), not a holistic-preference floor.

**Response-aware gating cost.** Candidates are decoded together with the unsteered reply (2 rows), costing about one
generation. Classifier scoring adds about 20–50 ms per reply. The real cost is that a steered turn cannot stream
token by token. E4 has the same cost. Latency is measured per condition in held-out.

---

## 3. Implementation map

| file | role |
|---|---|
| [chatbot_feature/emotion_reasoner/risk_screen.py](../chatbot_feature/emotion_reasoner/risk_screen.py) | steering-eligibility screen (EN / Hinglish / Devanagari; mention vs disclosure; history escalation) |
| [chatbot_feature/emotion_reasoner/selective.py](../chatbot_feature/emotion_reasoner/selective.py) | `SelectivePolicy` (JSON, fingerprint), `pre_decision`, `deficit_decision`, `accept`, `choose`, `language_of`, markers |
| [backend/selective_service.py](../backend/selective_service.py) | `SelectiveSteeringService` (`plan_selective`, `run_selective`), panel manifest and fingerprints, hook summary |
| [generation_analysis/e5_audit.py](../generation_analysis/e5_audit.py) | the audit in §1 |
| [generation_analysis/e5_data.py](../generation_analysis/e5_data.py) | Stage 1: supplement and held-out sets, leakage checks, manifest |
| [generation_analysis/e5_fit.py](../generation_analysis/e5_fit.py) | Stage 3: calibration procedure (§5) |
| [generation_analysis/e5_selective.py](../generation_analysis/e5_selective.py) | stage CLI: dev-generate/score, calibrate, freeze (pre-registration), generate, score |
| [generation_analysis/e5_report.py](../generation_analysis/e5_report.py), [e5_plots.py](../generation_analysis/e5_plots.py) | Stage 6: report, verdict, examples, figures |
| [data/e5_authored_messages.csv](../data/e5_authored_messages.csv) | 198 authored messages (Hindi, Hinglish, neutral, safety), dev/held-out split column |
| [tests/test_e5_selective.py](../tests/test_e5_selective.py) | 22 tests: screen, decisions, service (fake runtime), calibration, runner/resume, judge cache, report/verdict |

**Per-reply log** (`generations.jsonl`, condition E5, key `e5`):

- policy version and fingerprint;
- risk category, cues and source;
- language; eligibility stage and reason; cell, multipliers and planned alphas; E4's proposal;
- deficit (alignment, threshold, validation markers, decision);
- every candidate tried, with its check values and the rejection reason;
- applied alpha, panel and panel fingerprint;
- hooks: layers, neuron count, decode positions, `hooks_executed`, `shift_applied`;
- status, latency, generation and scoring seconds.

Neuron IDs per layer are written once per run (`heldout/<tag>/panel_manifest.json`) and referenced by fingerprint.

---

## 4. Data and leakage prevention

- **Development:** the 527 already-used messages (E3/E4 replies exist), plus a supplement drawn by `e5_data`. The
  supplement has self-criticism (EmpatheticDialogues *train*), mixed-valence GoEmotions (even rows), and authored
  Hindi (28), Hinglish (12), neutral (12) and indirect-risk (10) messages.
- **Held-out:** fresh. EmpatheticDialogues *test* (then validation) by situation category: positive, sadness,
  fear, anger, self-criticism, multi-turn. Mixed GoEmotions (odd rows). 40 dataset Hinglish plus 36 authored. 36
  authored Hindi. 24 neutral. Safety: dataset crisis cues plus authored explicit (8), indirect (18), other risks
  (6: abuse, medical, eating, substance), mentions and figurative (8), plus 7 dataset crisis messages.
- **Checks:** against every earlier split, study and held-out set (`steering_modes.used_messages` plus both studies'
  items) and the panel-discovery prompts (`results/*_generation_emotion/prompts/*.csv`):
  - exact match;
  - character 3–5-gram TF-IDF cosine ≥ 0.85;
  - MiniLM cosine ≥ 0.90.

  A drawn candidate that fails is skipped (listed in `leakage_excluded.csv`). An authored message that fails stops
  the build.
- **Flag:** every row of `data/mlt_hinghlish_dataset_clean.csv` was used for *recognition*-neuron selectivity (not
  for the steering panels), so dataset-Hinglish rows carry `discovery_overlap = recognition_corpus`.
- **Risk-screen lexicon.** It was extended from the *development* indirect-risk messages only: development recall
  3/10 before, 9/10 after; false flags 2 of 510 ordinary earlier messages. Held-out recall is measured, never tuned.

**Dry run, 2026-11-24.** Kept in memory, so nothing was frozen; §8 freezes it. Took 135 s.

Held-out set, **373 messages**:

| group | messages |
|---|---|
| English | 244: positive 28, sadness 28, fear 24, anger 16, self-criticism 24, multi-turn 40, mixed 30, neutral 16, plus English safety |
| Hinglish | 83: dataset 40, authored 36, neutral 4, plus Hinglish safety |
| Hindi | 46: authored 36, neutral 4, plus Hindi safety |
| Safety | 47: explicit 15 (8 authored + 7 dataset crisis cues), indirect 18, other 6, mention/figurative 8 |

Development supplement: **110 messages**.

Leakage: 290 drawn candidates were excluded. The worst remaining held-out message has character n-gram similarity
0.81 to an earlier text; held-out vs supplement, 0.61.

Dataset Hinglish lost 212 of 600 candidates as near-duplicates: the corpus is template-generated.

Unused GoEmotions crisis messages are exhausted. The crisis pool was then widened to EmpatheticDialogues
test/validation and the Hinglish corpus, which gave 7 dataset crisis messages.

**MiniLM is English-only.** Unrelated Devanagari sentences scored about 0.90, so the semantic check now covers
Latin-script text only.

The manifest records the risk screen's category counts on held-out (a composition field). The screen was not
changed after it was first computed on held-out data.

---

## 5. Calibration procedure (fixed before any calibration data)

`generation_analysis/e5_fit.py`, per model:

1. **Cells and estimates.** A cell is a (strategy, language) pair with at least 8 eligible messages; the remaining
   messages of each language form the residual cell `*|<language>`. For each cell and each option (one strength
   multiplier: 1.0, 0.75 or 0.5), the per-message change vs E3 is computed by replaying E5's steps 1–4 offline. The
   cell mean is shrunk toward the language mean of the same option, with a prior weight of 10 messages.
2. **Admissible options.** An option is admissible when all of these hold:
   - shrunk tone gain ≥ **0.15**;
   - each quality metric and validation ≥ **−0.05**;
   - safety ≥ **−0.05**;
   - raw harmed share ≤ **0.20**;
   - n ≥ **8**.

   The admissible option with the largest shrunk tone gain is chosen; if none is admissible, the cell is not steered.
3. **Population check.** Over **all** development messages, steered or not, the bootstrap 95% lower bound must be
   above −**0.10** for each quality metric and validation, and above −**0.05** for safety. If not, the steered cell
   contributing most to the safety (then quality) loss is dropped, and the check repeats.
4. **Global settings.** The deficit threshold {off, 0.30, 0.10} and word-ratio cap {off, 2.0} are chosen by 5-fold
   out-of-fold estimation over messages: the largest out-of-fold tone gain whose out-of-fold population check
   passes; ties go to the simpler setting.
5. **Refit and record.** The procedure is refitted on all development messages and written as a draft. The draft
   records every cell's evidence, the out-of-fold estimates, and a sensitivity analysis (minimum gain 0.10 / 0.20,
   **not** used for selection).

Coverage is never a target; it is whatever the constraints leave.

**Margins.** Quality 0.10 matches the confirmatory registration, so the two studies are comparable. The audit's
judge-noise table shows a systematic judge gap on mean differences ≤ 0.07 for three models (0.13–0.44 for Llama-2),
with per-message SDs of 0.3–0.9. Safety 0.05 is half the quality margin, because safety is never traded for tone.

**Preview with E4's 1.0× only** (existing data; the real fit adds 0.75× and 0.5×):

| model | steered contexts | development steered share | in-sample tone gain |
|---|---|---|---|
| Llama | grounding / English | 10% | +0.03 |
| EmoLLaMA | 6 cells | 37% | +0.18, quality +0.09 |
| Llama-2 | none | – | – |
| Sarvam | warm connection / Hinglish | 3% | – |

These previews motivate the 0.75× and 0.5× candidates in Stage 2. They are **not** the frozen policies.

---

## 6. Held-out evaluation and statistics

**Conditions on the same messages.** The first five use the chat's own code paths:

- E1: MindMate prompt, off.
- E2: MindMate prompt, always.
- E3: KG prompt, off.
- E4: KG prompt, always.
- E5-old: KG prompt, policy.
- E5: selective.

Reuse follows the earlier studies: Sarvam Hinglish/Hindi E3 = E1 (same prompt); E5-old = E3 when the policy does
not steer; E5 = E3 when the pre-generation gates decline.

**Judging.** Both judges, blind: system names never appear; pairwise preferences average both presentation orders.
Rubrics:

- tone toward the planned emotion;
- appropriateness, empathy, relevance, fluency, naturalness, safety;
- **validation** and **strategy adherence** (new);
- Hinglish or Hindi naturalness.

Automatic measures: language match, independent classifier, length, degenerate replies. Every answer is cached, so
an interrupted run resumes.

**Registered hypotheses** (per model; one-sided paired t; Holm over all models; written by `freeze` before any
held-out reply):

| id | contrast | test |
|---|---|---|
| H1 | E5 vs E3 | tone > 0 |
| H2 | E5 vs E3 | quality non-inferior (0.10) |
| H3 | E5 vs E3 | safety non-inferior (0.05) |
| H4 | E5 vs E4 | quality > 0 |
| H5 | E5 vs E4 | safety > 0 |
| H6 | E5 vs E5-old | tone > 0 |
| H7 | E5 vs E5-old | quality non-inferior |
| H8 | E5 vs E3 | validation non-inferior |

**Also reported:**

- E5 vs E1 and E5 vs E2;
- bootstrap CIs, Wilcoxon p, Cohen's dz, preference CIs, judge agreement;
- per-metric non-inferiority;
- attempt / retained / rejection rates, deficit skips, alpha distributions;
- latency, and the E5 overhead on attempted turns;
- harmed share;
- the batch-noise control;
- subgroups by language, category, safety category, strategy, target, user emotion and intensity, over the whole
  population and the eligible and steered subsets.

**Verdict rules** (`e5_report.verdict`): improves on E3 = H1 ∧ H2 ∧ H3. Reduces E4's trade-off = H4 ∨ H5. Improves
on E5-old = H6 ∧ H7. The recommendation follows from these:

- **enable:** improves on E3;
- **restrict:** tone confirmed without non-inferiority;
- **disable:** no tone gain, or the policy steers nothing.

E6 is judged *justified* only when steering helps tone but steered replies still lose quality or safety. The audit
showed lightweight checks cannot catch those replies.

---

## 7. Safety

- **The Safety Agent no longer exists in the working tree.** It was removed on 2026-11-22 at the user's request.
  Removed: `chatbot_feature/safety/` (incl. the draft validator `validator.py`), `agents/safety.py`, the LangGraph
  safety nodes, evaluation and tests. The backup is `../mindmate_safety_agent_backup_2026-11-22.tar.gz`.
- **What remains:** the Emotion Reasoner's crisis cue, which is negation-aware but narrow and switches the reply to
  the crisis-support strategy. Nothing checks a generated draft before it is sent.
- **Implication for E5.** E5 cannot give a safety agent authority over steering, because none exists. Instead, E5's
  safety step is a deliberately over-inclusive *eligibility* screen: steering stays off for crisis, disclosure,
  mention, indirect risk, abuse, harm to others and medical emergencies. It is not a response safety system and does
  not replace one.
- **What the screen does not do:**
  - check replies for inappropriate reassurance, diagnosis or dependency (only E5's acceptance step rejects a steered
    candidate that *adds* such phrasing);
  - escalate to helplines beyond the KG crisis strategy.
- **Live chat.** Mode `always` (the chat's mode) steers crisis messages toward warmth at the highest strength. That
  is E4's behaviour and was left unchanged.
- **Users and unvalidated candidates.** E5 runs only in the offline study; the chat API does not expose it. No user
  sees an unvalidated E5 candidate. Before E5 ever serves users, restoring a response-level safety check (the backed-up
  validator, or E6) is the user's decision.

---

## 8. Runbook

Run from `/data/mtech/surya_project/mental_health`. `results/` is gitignored, so frozen artifacts are committed with
`git add -f` (as the confirmatory registration was).

**GPU memory.**

- One model at a time (48 GB GPU): Llama 16 GB, EmoLLaMA 14 GB, Llama-2 26 GB, Sarvam 20 GB at 4-bit. Judges:
  Qwen-7B 15 GB, then Gemma-27B at 4-bit about 17 GB.
- Stop the chat server first.
- Generation and judging stages can be chained, but separate commands guarantee a clean GPU between them.
- Every long stage resumes: rerun the same command. `--retry-failed` re-attempts failed units.

```bash
cd /data/mtech/surya_project/mental_health && mkdir -p results/e5_selective/logs
```

**Stage 0 – pipeline smoke test** (optional; about 20–30 min). Uses 3 already-used messages and writes
`results/e5_selective/smoke/`. `--smoke-permissive` forces the candidate path.

```bash
tmux new -s e5smoke
python3 -m generation_analysis.e5_selective --model llama --smoke 3 --smoke-permissive \
  --stages dev-generate dev-score calibrate generate score report 2>&1 | tee results/e5_selective/logs/smoke_llama.log
```

**Stage 1 – audit, datasets, leakage** (CPU plus the small classifier; about 15–25 min). Review
`data/e5_authored_messages.csv` first if possible: the build freezes it.

```bash
python3 -m generation_analysis.e5_selective --stages audit build 2>&1 | tee results/e5_selective/logs/stage1.log
git add data/e5_authored_messages.csv && git add -f results/e5_selective/data && git commit -m "E5: freeze development supplement and held-out set"
```

Output:

- `results/e5_selective/audit/*`
- `results/e5_selective/data/{dev_supplement.csv, heldout.csv, manifest.json, leakage_excluded.csv}`

**Stage 2 – development generation and judging** (GPU, per model; roughly 1–2 h per model for Llama-family, longer
for Sarvam):

```bash
tmux new -s e5dev
for m in llama emollama llama2 sarvam; do
  python3 -m generation_analysis.e5_selective --model $m --stages dev-generate 2>&1 | tee -a results/e5_selective/logs/stage2_$m.log
  python3 -m generation_analysis.e5_selective --model $m --stages dev-score    2>&1 | tee -a results/e5_selective/logs/stage2_$m.log
done
# or detached:  nohup bash -c 'for m in llama emollama llama2 sarvam; do for s in dev-generate dev-score; do python3 -m generation_analysis.e5_selective --model $m --stages $s || exit 1; done; done' > results/e5_selective/logs/stage2.log 2>&1 &
```

Output: `results/e5_selective/dev/<tag>/{generations.jsonl, lock.json, dev_replies.csv, dev_ratings_judge1.csv,
dev_ratings_judge2.csv, judge_cache_judge*.jsonl}`.

**Stage 3 – calibration** (CPU, minutes), then **send the results for analysis before freezing**:

```bash
for m in llama emollama llama2 sarvam; do python3 -m generation_analysis.e5_selective --model $m --stages calibrate; done 2>&1 | tee results/e5_selective/logs/stage3.log
```

Output:

- `results/e5_selective/policies/<tag>.draft.json`
- `results/e5_selective/calibration/<tag>_report.json`, `<tag>_cells.csv`, `<tag>_table.csv`

After the review, freeze and register. This refuses if held-out replies exist, then commits the registration:

```bash
python3 -m generation_analysis.e5_selective --stages freeze
git add -f results/e5_selective/policies results/e5_selective/preregistration.* && git commit -m "E5: freeze policies, pre-register hypotheses"
```

**Stage 4 – held-out generation** (GPU, per model; 6 conditions × 373 messages; refuses if any registered file
changed):

```bash
tmux new -s e5gen
for m in llama emollama llama2 sarvam; do
  python3 -m generation_analysis.e5_selective --model $m --stages generate 2>&1 | tee -a results/e5_selective/logs/stage4_$m.log
done
```

Output: `results/e5_selective/heldout/<tag>/{generations.jsonl, lock.json, panel_manifest.json}`.

**Stage 5 – judges** (GPU, per model):

```bash
for m in llama emollama llama2 sarvam; do
  python3 -m generation_analysis.e5_selective --model $m --stages score 2>&1 | tee -a results/e5_selective/logs/stage5_$m.log
done
```

Output: `heldout/<tag>/{replies.csv, ratings_judge1.csv, ratings_judge2.csv, preferences_judge1.csv,
preferences_judge2.csv, judge_cache_*.jsonl}`.

**Stage 6 – analysis** (CPU):

```bash
for m in llama emollama llama2 sarvam; do python3 -m generation_analysis.e5_selective --model $m --stages report; done
python3 -m generation_analysis.e5_selective --stages verdict plots examples 2>&1 | tee results/e5_selective/logs/stage6.log
```

Output:

- `heldout/<tag>/{summary.json, condition_means.csv, contrasts.csv, subgroups.csv, coverage.json, latency.csv}`
- `results/e5_selective/{report.md, report.json, examples.md}`
- `results/e5_selective/plots/{coverage,tradeoff,alpha}.{png,csv}`

**Monitoring:**

```bash
tail -f results/e5_selective/logs/stage2_llama.log             # live log
cat results/e5_selective/progress/*.json                         # done / total / per minute / ETA / failures per stage
tail -n 60 results/e5_selective/logs/errors.log                  # tracebacks of failed units
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 15
tmux attach -t e5dev                                             # reattach after disconnecting
```

---

## 9. Limitations

- **Judges.** Judges agree only moderately on per-message *quality* differences: Spearman 0.25–0.56, sign agreement
  0.61–0.79. Small quality differences are within judge noise, which is why non-inferiority margins are used rather
  than "not significant".
- **Classifier as deficit signal.** It is weak on Hinglish, so the deficit gate is English-only. Hindi Devanagari
  reliability is unknown.
- **Authored messages.** The Hindi, part of the Hinglish, the neutral and the safety messages were written by an AI
  assistant, not native speakers or clinicians. Dataset Hinglish is machine-generated.
- **Risk screen.** It is lexical. Held-out indirect-risk recall will show its gaps; a missed indirect-risk message
  can be steered if its context is admitted.
- **Batch noise.** Batched vs single-row decoding noise is substantial for Sarvam. Effects smaller than the measured
  noise floor should not be interpreted.
- **E5 vs E4 is not a pure test of selection.** E5 can only reduce E4's coverage; it never adds steering E4 lacks.
- **E6.** No response critic is implemented. Its justification is a registered output of this study, not an
  assumption.
