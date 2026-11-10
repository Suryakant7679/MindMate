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
