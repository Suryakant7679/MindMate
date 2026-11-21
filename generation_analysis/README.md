# Generation-time emotion neurons (Llama 3.1 8B Instruct)

Extends the recognition analysis — dataset-text forward passes, input to
`layer.mlp.down_proj`, 32 × 14,336 neurons, `S = P·(1−H)` over ten labels —
to autoregressive response generation. It uses the original
`meta-llama/Llama-3.1-8B-Instruct` checkpoint. LoRA adapters and other
checkpoints are refused unless you pass `--allow-model-mismatch`.

The recognition outputs (`results/llama31_8b_emotions_full`,
`results/llama31_8b_neuron_analysis`) are only read. Everything this package
produces goes under `results/llama31_8b_generation_emotion/`.

```sh
bash generation_analysis/run.sh            # full pipeline (GPU, ~16 GiB for bf16 weights)
bash generation_analysis/run.sh --resume   # continue an interrupted run
python3 -m unittest discover -s generation_analysis/tests -t . -v
```

## Design

**Matched responses.** Neutral-labelled dataset messages are the user
content. Each message is answered ten times, once per recognition label
(neutral included). The ten prompts differ only in one style sentence of the
system prompt, so every emotional response has a neutral twin with the same
user content. Identification prompts (default 200 → 2,000 responses) and
held-out steering prompts (default 32) do not overlap.

**Prefill vs decoding.** `common.greedy_decode` runs prefill and each KV-cache
decode step as separate forward calls and labels the phase before each call.
Phases are therefore never inferred from tensor shapes.

- *Prefill*: all non-special prompt positions.
- *Decode step s*: reads generated token `y_{s-1}` and emits `y_s`. It is
  stored in slot `s−1`. Only non-special read tokens are recorded.

For token-level analysis, `--token-alignment emit` (default) pairs `y_i` with
slot `i−1`, the step that produced it, and `read` pairs it with slot `i`. The
first decode-batch is re-run as one teacher-forced pass and compared
position-by-position (`activations/alignment_check.json`). The unit tests
assert that the two passes agree exactly and that batched left-padded decoding
equals unbatched decoding.

**Neuron sets** (`identify_neurons`, top percentage per layer, same formula and
tie-breaking as `emotion_neuron_extraction.py`):

| set | source |
|---|---|
| recognition `R_e` | `results/llama31_8b_emotions_full/selectivity.npy` |
| generation `G_e` | decode-step firing grouped by response condition (ten labels) |
| emotional-token `E_e` | decode slots of emotional words in condition-`e` responses |
| semantic `Sem` | decode slots of shared, non-emotional content words |

The token classes `emotional_<e>` (nine) and `semantic` are scored jointly
with `S = P·(1−H)`. A generated word is **emotional** if it matches the
emotion lexicon or its matched-prompt log-ratio
`Σ log p(word | emotion prompt) − log p(word | neutral prompt)` (identical
prefix) is at least 1 nat. A word is **semantic** if it is a non-emotion
content word, its log-ratio is within ±0.5 nats, and it also occurs in the
user message or the neutral twin. By default, semantic tokens come from the
same emotional responses as the emotional tokens.

`--adherence matched` (default) keeps an emotional response only if the frozen
recognition probe gives its emotion a higher probability than for the neutral
twin. Retention rates are in `token_analysis/adherence.csv`.

**Overlap.** `overlap/layerwise_jaccard.csv` gives, for every percentage, layer,
emotion and pair of {R, G, E, Sem}: sizes, intersection, Jaccard, the Jaccard
expected for random sets of equal size, enrichment, and a hypergeometric
p-value. `union_of_emotions` rows compare the union over non-neutral emotions.

**Steering** (`steering`). Held-out messages are answered under the neutral
prompt. One set's neurons receive an additive shift on every decode step
(prefill unchanged, as in the backend). The default shift is
`alpha·(μ_e − μ_neutral)` from decode means, so every set uses the same
statistics and only the neuron choice differs. Each set's shift is rescaled to
the generation set's L2 norm (`--delta-norm matched`). Sets: recognition,
generation, emotional-token, and a layer-matched random control.
`--delta-mode selectivity` uses the backend's panel rule instead.

Metrics per response, relative to the unsteered response for the same prompt:

- probe target-emotion probability change and argmax hit
- target-lexicon rate
- MiniLM embedding similarity
- content-word F1
- user-content recall
- NLL change under the unsteered model
- repeated-bigram rate

`paired_comparisons.csv` holds Wilcoxon signed-rank tests paired by
(emotion, prompt).

## Chatbot steering panels (`build_steering_panels`, `evaluate_chat_steering`)

The selectivity-based sets above respond to an emotion but steering them does
not change the emotion of generated text (`steering/steering_summary.csv`).
The chatbot therefore uses panels chosen by what neurons *write*:

```sh
python3 -m generation_analysis.build_steering_panels      # writes steering_panels/panels.json
python3 -m generation_analysis.evaluate_chat_steering     # steered vs unsteered vs random neurons
```

For each emotion and layer, `d` is the mean residual-stream difference between
emotion-styled and neutral-styled replies (teacher-forced over the saved
matched responses). A neuron with output weight `w` and decode-step mean
activation shift `dmu` scores `dmu * (w . d/|d|)`. The top 1% per layer in
layers 8-18 form the panel, and each neuron stores its `dmu`. The backend adds
`alpha * STEERING_GAIN * dmu` (`backend/config.py`) to those neurons on every
decode step. `evaluate_chat_steering` runs the real backend path with the
MindMate prompt and writes `chat_steering_eval/{replies,summary}.csv`.

## Steering policy calibration (`steering_policy_experiment`, `calibrate_steering_policy`, `validate_steering_policy`)

`steering_policy_experiment` answers 42 gold-labelled messages under four
conditions (baseline, KG strategy prompt, steering, KG prompt + steering) at
six strengths with greedy decoding, and scores every reply with independent
models: target emotion by the GoEmotions XLM-R classifier; empathy, contextual
relevance and fluency by Qwen2.5-7B-Instruct rubric judgements; semantic
preservation by MiniLM similarity to the unsteered reply. It also records how
often the Llama probe, the classifier and the KG-fused detector agree with the
gold user emotion. `calibrate_steering_policy` selects, per context, the
strength with the largest gain among those that did not lower judged quality
and writes `steering_policy/policy.json`, which the Emotion Reasoner loads.
`validate_steering_policy` runs the deployed service on 24 held-out messages.

## Causal validation of R∩G, R\G and G\R (`causal_validation`)

```sh
HF_HUB_OFFLINE=1 python3 -m generation_analysis.causal_validation           # ~1 h on one GPU
python3 -m generation_analysis.causal_validation --scopes all bands layers   # adds 32 single-layer runs per group
```

This uses the existing R and G files (`neuron_sets.json`, top 1%) and does not
change how neurons are selected. For each emotion and layer it builds:

- `shared` R∩G
- `recognition_only` R\G
- `generation_only` G\R
- `random`: neurons drawn from outside R∪G in the same layer

With `--size-matching matched` (the default), every group has |R∩G| neurons
in each layer. R\G and G\R are subsampled with a fixed seed. Group sizes per
layer are in `causal_validation/groups/`.

**Masking** is mean-ablation: a masked neuron is set to its natural mean, so
random neurons are never pushed outside their own range. The realised change
(mean per-position L2 norm) is reported for every condition. Special tokens
and padding are never modified.

**Tasks**, run separately:

- **Recognition:** the frozen probe on 60 probe-test texts per class, masked
  at all positions. The reference mean comes from probe-train texts. Metrics:
  target-emotion probability, target recall, accuracy, macro-F1 and
  target F1.
- **Generation:** emotion-prompted replies to the 32 held-out prompts, masked
  on decode steps only (`--generation-positions all` also masks the prompt).
  Replies are scored with `suryakantmani/xlm-roberta-large-goemotions-hard`,
  which scores accuracy 0.68 and macro-F1 0.68 on the 2,000 matched replies,
  against 0.40 accuracy for the Llama probe. Metrics: target-emotion
  probability, NLL change, MiniLM similarity to the unmasked reply,
  repeated-bigram rate, changed-reply rate, and token change rate (1 minus the
  shared prefix fraction).

**Positive controls** shift every neuron by the full emotion-vs-neutral mean
difference. `pc_remove` subtracts it (recognition texts and emotion-prompted
replies), and `pc_add` adds it to neutral-prompted replies. A control "works"
for an emotion if its mean effect is at least 0.05 and Wilcoxon p < 0.05.
Group effects are normalised by the `pc_remove` effect only for emotions
whose control works. `positive_control.json` states whether the generation
setup is insufficient.

**Reports:**

- `effects_by_group_and_scope.csv`: by group type and layer scope
  (all layers and four 8-layer bands)
- `effects_by_emotion.csv`: by emotion
- `{recognition,generation}/summary_by_emotion.csv`: all task metrics
- `hypothesis_tests.csv`: group vs. random (paired Wilcoxon), and
  specificity = normalised recognition excess minus normalised generation
  excess
- `effects_by_group.png`

## Outputs (`results/llama31_8b_generation_emotion/`)

| path | content |
|---|---|
| `prompts/` | identification and held-out steering prompts |
| `activations/responses.jsonl` | responses, token ids, log-probs, phase counts |
| `activations/{decode,prefill}_{firing_counts,activation_mean}.npy` | per response `[N, 32, 14336]` |
| `activations/decode_token_firing_bits.npy` | bit-packed firing per decode slot `[N, M, 32, 1792]` |
| `token_analysis/` | token log-ratios, probe scores, token categories, adherence |
| `neuron_sets/` | selectivity tensors, condition means, `neuron_sets.json`, per-percentage CSVs |
| `overlap/` | layer-wise Jaccard table, summary, plots |
| `steering/` | responses, scores, summaries with bootstrap CIs, paired tests, trade-off plot |

At the defaults (2,000 responses, 64 new tokens), activation arrays take about
15 GB.

## Causal panel validation, recalibration and held-out comparison (`causal_panel_experiment`, `recalibrate_policy`, `heldout_comparison`)

```sh
HF_HUB_OFFLINE=1 python3 -m generation_analysis.causal_panel_experiment        # dev split: attribution, likelihood, interventions
HF_HUB_OFFLINE=1 python3 -m generation_analysis.recalibrate_policy --panels deployed cand:deployed_causal_half cand:grad_global_half
HF_HUB_OFFLINE=1 python3 -m generation_analysis.heldout_comparison --stages v1 v2 score report   # test split, read once
python3 -m generation_analysis.export_steering_evidence --variant grad_global_half               # Mood Tracker evidence
```

Splits (`causal_steering/splits/`, frozen): dev = EmpatheticDialogues train; calibration = validation + Hinglish even
rows; test = test + Hinglish odd rows. Evaluators the run-time gate never uses: Qwen2.5-7B 1-5 ratings (emotion,
empathy, relevance, fluency, pairwise preference) and `SamLowe/roberta-base-go_emotions`. Uncertainty: paired
bootstrap CIs over messages, Wilcoxon tests, Holm correction on the primary contrasts. `deployed_v1_service.py` and
`steering_policy/policy_v1_deployed.json` freeze the previous system. Active panel file: `backend/config.py`
`STEERING_PANEL_FILE` (`panels_causal.json`; set it back to `panels.json` to revert the panels).
Results: `results/llama31_8b_generation_emotion/causal_steering/{panel_causality,recalibration,heldout}`.

## Two-judge consensus study (Llama 3.1-8B and EmoLLaMA-chat-7B)

Experimental; nothing here changes what the chatbot does.

```bash
python3 -m generation_analysis.freeze_v2                 # once: freeze deployed panels, policy, results
python3 -m generation_analysis.freeze_v2 --verify        # deployed files still identical to the freeze?
for m in llama emollama; do
  MINDMATE_MODEL=$m python3 -m generation_analysis.diagnose_calibration   # why contexts were rejected
  MINDMATE_MODEL=$m python3 -m generation_analysis.consensus_policy       # experimental policy (calibration only)
done
python3 -m generation_analysis.holdout2 --stages build   # new held-out messages, shared by both models
for m in llama emollama; do
  MINDMATE_MODEL=$m python3 -m generation_analysis.holdout2 --stages generate score report
done
python3 -m generation_analysis.holdout2 --stages compare
```

- `results/frozen_v2/<tag>/` holds the frozen v2 configuration and a SHA-256 manifest. Every later
  stage verifies it first.
- The consensus policy keeps the deployed selection rule but requires both judges (Qwen2.5-7B,
  Gemma 3-27B) to pass it separately, with the independent GoEmotions classifier rising. It is
  written to `causal_steering/consensus/policy_consensus.json` and is not installed.
- `results/steering_holdout2/items.csv` contains messages that are in no earlier split (checked by
  conversation id and text). Policies are hashed when generation starts (`policy_lock.json`); a
  changed policy is refused.
- Each model is steered only with its own panels. Results per model are in
  `causal_steering/holdout2/` (`system_means.csv`, `contrasts.csv`, `safety.csv`,
  `judge_agreement.csv`, `decision_counts.csv`); the cross-model table is
  `results/steering_holdout2/model_comparison.csv`.

## Version-3 generation-quality study (Llama 3.1-8B, EmoLLaMA-chat-7B, Sarvam-30B)

Experiments only: the deployed prompt, frozen v2 policies, agents, safety path and Mood Tracker are unchanged.

```bash
python3 download_half_precision.py sarvamai/sarvam-30b          # one-off: 128 GB float32 -> 60 GB bf16 on disk
python3 -m generation_analysis.v3_common                         # extra Hinglish dev/calibration messages, holdout3
for m in llama emollama sarvam; do                               # prompt ablation on calibration messages
  MINDMATE_MODEL=$m python3 -m generation_analysis.v3_prompts --stages generate score report  # sarvam: add --think off
done
for m in llama emollama; do                                      # steering methods: screen (dev), calibrate, select
  MINDMATE_MODEL=$m python3 -m generation_analysis.v3_methods --stages screen screen_score screen_report --prompt kg_v2
  MINDMATE_MODEL=$m python3 -m generation_analysis.v3_methods --stages calibrate calibrate_score select --prompt kg_v2
done
# write results/steering_v3/decisions.json from the calibration tables, then (read once):
MINDMATE_MODEL=<model> python3 -m generation_analysis.holdout3 --stages generate score report
python3 -m generation_analysis.holdout3 --stages compare
```

- `chatbot_feature/emotion_reasoner/tone.py`: version-3 prompt (per-strategy tone, style and Hinglish rules, degenerate-reply
  check). `steering_methods.py`: additive, bounded, band-selective and contrastive-residual interventions.
- `holdout3` messages and the decisions file are hashed when generation starts (`lock.json`); a change is refused.
- Sarvam's reasoning is skipped by pre-filling an empty reasoning block (`generate_reply(..., think=False)`), because the
  checkpoint ignores its own no-thinking flag and otherwise spends about 1,300 tokens (~50 s) per reply.

## Sarvam-30B Hinglish emotion-neuron steering (`sarvam_lab.py`, `sarvam_experiments.py`)

Sarvam's own panels (never transferred from Llama): layer 0 dense MLP (8,192 neurons) and the shared expert of
layers 1-18 (1,024 each) are the always-active sites; the 128 routed experts are too sparse to steer. Panels are
discovered from 170 matched prompts (120 Hinglish): paired |t| >= 3, then top attribution per site (1,328 neurons
per tone). A residual-stream direction after layer 10 is the second intervention. Reasoning is off in every run.

```bash
MINDMATE_MODEL=llama python3 -m generation_analysis.sarvam_lab data          # message sets (dev / calibration / held-out)
python3 -m generation_analysis.sarvam_lab discover panels                     # activations, panels.json, residual directions
A="--alphas 1 2 3 4 --language-alphas 0.5 0.75 1.0 1.25 --combo-language 0.75"
python3 -m generation_analysis.sarvam_experiments causal causal_score causal_report $A      # dev, judge 1
python3 -m generation_analysis.sarvam_experiments causal2 $A                                # residual directions (dev)
python3 -m generation_analysis.sarvam_experiments causal3 causal3_score causal3_report $A   # dev, both judges: sham,
#   wrong-panel, third random draw, top-k / random subsets, concentrated top-10, random and reversed directions, ablations
python3 -m generation_analysis.sarvam_experiments calibrate calibrate_score compare select $A   # both judges, policy
python3 -m generation_analysis.sarvam_experiments heldout heldout_score heldout_report $A       # untouched, via the real service
python3 -m generation_analysis.sarvam_experiments heldout_language heldout_language_score heldout_language_report $A
python3 -m generation_analysis.sarvam_experiments smoke evidence                             # real-model trace check, evidence
python3 -m generation_analysis.sarvam_experiments install                                    # refuses unsupported steering
python3 tests/e2e_sarvam_chat.py                                                             # HTTP end-to-end chat test
```

- Results: `results/sarvam_30b_generation_emotion/steering/` (`causal*/`, `calibrate/`, `heldout/`, `heldout_language/`,
  `evidence.json`, `policy_candidate.json`); panels in `steering_panels/panels.json`.
- `install` copies the policy only when every steering cell is backed by the held-out test (tone gain CI > 0 for both
  judges, no quality CI below -0.15); a policy that steers nothing is installed as an explicit "steering disabled"
  state so the chat service never falls back to heuristic strengths on unvalidated panels.
- Held-out files are locked by hash (`lock.json`); the policy is never changed from held-out outcomes.

## Emotion & neuron steering analytics (Mood Tracker dashboard)

The Mood Tracker has six sections: Overview, Emotion Timeline, Response Comparison, Neuron Explorer, Causal
Ablations and Model Evaluation. Every number is labelled Measured, Classifier-estimated, LLM-judged, User-reported
or Not available. Opening the dashboard runs no model: it reads per-turn records and saved experiment files.

```bash
python3 -m generation_analysis.neuron_validation --model <llama|emollama|sarvam> all   # causal battery (GPU, both judges)
python3 -m generation_analysis.neuron_validation --model <model> analyse               # re-write report.json and cards.json (CPU)
python3 -m generation_analysis.neuron_validation summary                               # cross-model table
python3 -m generation_analysis.trajectory_validation                                   # emotion detector vs labelled messages
python3 -m backend.server --model llama --research-compare                             # chat; turns are recorded in results/telemetry/
python3 -m generation_analysis.turn_evaluation --model llama                           # judge recorded steered/unsteered replies
node tests/browser/dashboard_check.js http://127.0.0.1:8080                            # browser check against the backend records
```

- API (read-only, saved data): `/api/emotion/trace/<turn>`, `/api/emotion/conversation/<id>`, `/api/emotion/neurons/<turn>`,
  `/api/emotion/panel/<name>`, `/api/emotion/research`, `/api/emotion/models`; `POST /api/emotion/feedback` (voluntary 1-5
  ratings) and `POST /api/emotion/settings` (research comparison on or off).
- Research comparison generates a steered candidate on turns the policy does not steer, in the same decoding batch; it is
  never sent and never generated on a safety turn. API strength overrides need `--allow-overrides` and still pass the
  reply safeguards.
- The battery establishes panel-level effects only (panel, layer bands, ranked subsets, each against matched random
  neurons, sham hooks and the reversed shift). No single neuron is tested, and the dashboard says so.
- `MINDMATE_VALIDATION_HINGLISH=1` adds the 24 Hinglish development messages to the Llama-family battery
  (then re-run `generate score --force analyse`).

### Single-neuron interventions (`single_neuron.py`)

Needs the model's battery. Stage 1 screens the 20 top-ranked neurons of each panel and 20 matched control neurons with
teacher-forced log-likelihoods (one neuron alone at 1x, 8x, 32x the reference strength; the panel without the neuron).
Stage 2 generates and judges replies for the 5 best-screened neurons and 5 controls. A neuron is "supported" only when
the screen passes (CI above zero, Holm, above the controls' 95th percentile) and the judged emotion rises beyond controls.

```bash
python3 -m generation_analysis.single_neuron --model <llama|emollama|sarvam> all     # screen, generate, score, analyse
```
Results: `neuron_validation/single/report.json`; shown in the Neuron Explorer (panel explorer and evidence ladder).
