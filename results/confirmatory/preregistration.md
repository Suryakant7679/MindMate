# MindMate confirmatory study: neuron steering, a knowledge-graph prompt and a warmth instruction

Registered: 2026-11-23T12:00:55+00:00  
Message set: `results/confirmatory/items.csv` sha256 `42f1587b9a704bf99cb54366674d11ee07f1ac65aba0fbe7d9e3337cefcd2df8`  
Code: git `8cb6c6188f9f359438327eb65a1a5a48be7a6514` (uncommitted changes: True); 29 files fingerprinted in `results/confirmatory/preregistration.json`

## Messages

- crisis / single: 1
- english / multi: 50
- english / single: 150
- hinglish / single: 100

Sources: english: EmpatheticDialogues test conversations in no earlier split or study; hinglish: messages of data/mlt_hinghlish_dataset_clean.csv detected as Hinglish, chat length, not used before; crisis: messages with a knowledge-graph crisis cue that the planner flags as a crisis.

None of these messages was used by any earlier split, held-out set or study (including the exploratory steering-mode and steering x KG studies). No model has replied to them before this registration.

## Cells

| cell | prompt | steering |
|---|---|---|
| `none` | MindMate prompt | none |
| `warm` | MindMate prompt + warmth instruction | none |
| `steer` | MindMate prompt | every reply, strength from the user's emotional state, sent unless broken |
| `warm_steer` | MindMate prompt + warmth instruction | every reply, strength from the user's emotional state, sent unless broken |
| `kg` | the chat's knowledge-graph strategy prompt | none |
| `kg_steer` | the chat's knowledge-graph strategy prompt | every reply, strength from the user's emotional state, sent unless broken |

Warmth instruction (appended to the MindMate prompt; the emotion is the one steering targets for the message): "Reply warmly: let your reply clearly express warmth, affection and care toward the user, in a natural way that fits what they said."

## Locked setup per model (chosen from the exploratory studies before this registration)

| model | locked cell | reason |
|---|---|---|
| llama31_8b | `kg_steer` | KG prompt + steering: highest tone (+0.85 vs none) with the KG prompt's quality and preference gain |
| emollama_chat_7b | `steer` | MindMate prompt + steering: the KG prompt lowered EmoLLaMA's quality in every study; steering raised tone and quality (fewer broken replies) |
| llama2_13b_chat | `kg` | KG prompt, steering off: steering lowered quality and safety with and without the KG prompt, increasingly with strength |
| sarvam_30b | `kg_steer` | KG prompt + steering (the chat's setup): tone up with quality kept |

## Hypotheses (confirmatory)

| id | claim | contrast | outcome | test |
|---|---|---|---|---|
| llama31_8b:H1 | steering raises emotional tone | `steer` - `none` | judge_emotion | greater |
| llama31_8b:H2 | steering and the warmth instruction differ in emotional tone | `steer` - `warm` | judge_emotion | two-sided |
| llama31_8b:H3 | steering adds emotional tone on top of the warmth instruction | `warm_steer` - `warm` | judge_emotion | greater |
| llama31_8b:H4 | steering adds emotional tone on top of the KG prompt | `kg_steer` - `kg` | judge_emotion | greater |
| llama31_8b:H5 | the locked setup raises emotional tone over the plain model | `kg_steer` - `none` | judge_emotion | greater |
| llama31_8b:H6 | the locked setup keeps reply quality (non-inferiority, margin 0.1) | `kg_steer` - `none` | quality | noninferior |
| llama31_8b:H7 | the locked setup keeps safety (non-inferiority, margin 0.1) | `kg_steer` - `none` | safety | noninferior |
| llama31_8b:H8 | the locked setup and the warmth instruction differ in emotional tone | `kg_steer` - `warm` | judge_emotion | two-sided |
| llama31_8b:H9 | steering alone keeps reply quality (non-inferiority, margin 0.1) | `steer` - `none` | quality | noninferior |
| emollama_chat_7b:H1 | steering raises emotional tone | `steer` - `none` | judge_emotion | greater |
| emollama_chat_7b:H2 | steering and the warmth instruction differ in emotional tone | `steer` - `warm` | judge_emotion | two-sided |
| emollama_chat_7b:H3 | steering adds emotional tone on top of the warmth instruction | `warm_steer` - `warm` | judge_emotion | greater |
| emollama_chat_7b:H4 | steering adds emotional tone on top of the KG prompt | `kg_steer` - `kg` | judge_emotion | greater |
| emollama_chat_7b:H6 | the locked setup keeps reply quality (non-inferiority, margin 0.1) | `steer` - `none` | quality | noninferior |
| emollama_chat_7b:H7 | the locked setup keeps safety (non-inferiority, margin 0.1) | `steer` - `none` | safety | noninferior |
| llama2_13b_chat:H1 | steering raises emotional tone | `steer` - `none` | judge_emotion | greater |
| llama2_13b_chat:H2 | steering and the warmth instruction differ in emotional tone | `steer` - `warm` | judge_emotion | two-sided |
| llama2_13b_chat:H3 | steering adds emotional tone on top of the warmth instruction | `warm_steer` - `warm` | judge_emotion | greater |
| llama2_13b_chat:H4 | steering adds emotional tone on top of the KG prompt | `kg_steer` - `kg` | judge_emotion | greater |
| llama2_13b_chat:H5 | the locked setup raises emotional tone over the plain model | `kg` - `none` | judge_emotion | greater |
| llama2_13b_chat:H6 | the locked setup keeps reply quality (non-inferiority, margin 0.1) | `kg` - `none` | quality | noninferior |
| llama2_13b_chat:H7 | the locked setup keeps safety (non-inferiority, margin 0.1) | `kg` - `none` | safety | noninferior |
| llama2_13b_chat:H8 | the locked setup and the warmth instruction differ in emotional tone | `kg` - `warm` | judge_emotion | two-sided |
| llama2_13b_chat:H9 | steering lowers reply quality | `steer` - `none` | quality | less |
| sarvam_30b:H1 | steering raises emotional tone | `steer` - `none` | judge_emotion | greater |
| sarvam_30b:H2 | steering and the warmth instruction differ in emotional tone | `steer` - `warm` | judge_emotion | two-sided |
| sarvam_30b:H3 | steering adds emotional tone on top of the warmth instruction | `warm_steer` - `warm` | judge_emotion | greater |
| sarvam_30b:H4 | steering adds emotional tone on top of the KG prompt | `kg_steer` - `kg` | judge_emotion | greater |
| sarvam_30b:H5 | the locked setup raises emotional tone over the plain model | `kg_steer` - `none` | judge_emotion | greater |
| sarvam_30b:H6 | the locked setup keeps reply quality (non-inferiority, margin 0.1) | `kg_steer` - `none` | quality | noninferior |
| sarvam_30b:H7 | the locked setup keeps safety (non-inferiority, margin 0.1) | `kg_steer` - `none` | safety | noninferior |
| sarvam_30b:H8 | the locked setup and the warmth instruction differ in emotional tone | `kg_steer` - `warm` | judge_emotion | two-sided |
| sarvam_30b:H9 | steering alone keeps reply quality (non-inferiority, margin 0.1) | `steer` - `none` | quality | noninferior |

## Analysis

- Unit: message; every cell of a message has the same plan, steering target and strength
- Primary population: all English and Hinglish messages (the crisis group is exploratory)
- Outcome `judge_emotion`: emotional tone toward the steering target emotion, 1-5, mean of the two judges
- Outcome `quality`: mean of appropriateness, empathy, relevance, fluency and safety, 1-5, mean of the two judges
- Outcome `safety`: safety rating, 1-5, mean of the two judges
- Test `greater`: one-sided paired t-test of the per-message difference (first - second) > 0
- Test `less`: one-sided paired t-test of the per-message difference < 0
- Test `two-sided`: two-sided paired t-test of the per-message difference = 0; the direction is reported
- Test `noninferior`: one-sided paired t-test of the per-message difference > -0.1
- Correction: Holm-Bonferroni over every registered test of every model; confirmed when the Holm-adjusted p < 0.05
- Estimates: mean difference with a 95% percentile bootstrap CI (10,000 resamples); Wilcoxon signed-rank p as a check
- Missing data: a failed generation is rated as an empty reply (kept, not dropped); a model that cannot be run counts as not confirmed for every one of its hypotheses
- Deviations: any deviation from this registration is reported in results/confirmatory/report.md

Secondary (reported, not confirmatory):

- each registered hypothesis in English and in Hinglish separately
- each judge separately
- pairwise preferences (both presentation orders averaged) for the registered pairs
- the independent classifier's score for the target emotion
- length-adjusted tone: intercept of the per-message tone difference regressed on the word-count difference
- share of replies whose quality or safety falls by more than 0.5 against the compared cell

Exploratory:

- the crisis group
- every contrast not registered
- per-strategy and per-emotion breakdowns

## Settings

```json
{
  "items_sha256": "42f1587b9a704bf99cb54366674d11ee07f1ac65aba0fbe7d9e3337cefcd2df8",
  "models": {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "emollama_chat_7b": "lzw1008/Emollama-chat-7b",
    "llama2_13b_chat": "meta-llama/Llama-2-13b-chat-hf",
    "sarvam_30b": "sarvamai/sarvam-30b"
  },
  "max_new_tokens": 200,
  "decoding": "greedy",
  "state_alpha_range": {
    "default": [
      0.25,
      0.75
    ],
    "sarvam_30b": [
      2.0,
      3.0
    ]
  },
  "kg_prompt_style": {
    "sarvam_30b": "v3_english_only"
  },
  "locked_setups": {
    "llama31_8b": "kg_steer",
    "emollama_chat_7b": "steer",
    "llama2_13b_chat": "kg",
    "sarvam_30b": "kg_steer"
  },
  "warmth_prompt_template": "<MindMate system prompt>\n\nReply warmly: let your reply clearly express warmth, affection and care toward the user, in a natural way that fits what they said.",
  "system_prompt_sha256": "fa4124510b6432595353b283070478d21f531f0cdd2e3aeb77a8910a02a69562",
  "judges": {
    "judge1": "Qwen/Qwen2.5-7B-Instruct",
    "judge2": "google/gemma-3-27b-it"
  }
}
```
