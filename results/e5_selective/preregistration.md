# MindMate E5: quality-constrained adaptive selective emotion-neuron steering

Registered: 2026-11-24T15:31:25+00:00  
Held-out set sha256 `a8f9750d7efc484736901a3f0bd7fbda6b6a07815dca4a2dede319d6de3c5053`; git `8be0b7e0788220c92a398cb943f117bbefc920f0` (uncommitted changes: True); 42 files fingerprinted.

## Conditions

| condition | prompt | steering |
|---|---|---|
| E1 | MindMate prompt | off |
| E2 | MindMate prompt | always |
| E3 | the chat's KG strategy prompt | off |
| E4 | the chat's KG strategy prompt | always |
| E5old | the chat's KG strategy prompt | policy |
| E5 | the chat's KG strategy prompt | selective |

## Frozen E5 policies

| model | steered cells (strength multipliers) | deficit gate | sha256 |
|---|---|---|---|
| llama31_8b | *|hindi x0.75, strategy:appreciative_mirroring|english x0.75, strategy:gentle_exploration|hindi x0.75, strategy:grounding_reassurance|english x1.0, strategy:warm_connection|hinglish x0.5 | off | `2b10a9ebb62e` |
| emollama_chat_7b | strategy:appreciative_mirroring|english x1.0, strategy:compassionate_comfort|english x0.75, strategy:compassionate_comfort|hinglish x1.0, strategy:gentle_exploration|hinglish x1.0, strategy:grounding_reassurance|hinglish x1.0, strategy:reflective_validation|english x0.75, strategy:strengths_affirmation|english x1.0, strategy:warm_connection|hinglish x1.0 | off | `9216c8813f1c` |
| llama2_13b_chat | none | off | `55fd18d4c515` |
| sarvam_30b | *|hindi x0.5, strategy:appreciative_mirroring|hinglish x0.5, strategy:gentle_exploration|hindi x0.5, strategy:warm_connection|hinglish x1.0 | off | `3ed99682a27a` |

## Hypotheses

| id | claim | contrast | outcome | test |
|---|---|---|---|---|
| llama31_8b:H1 | E5 raises emotional tone over KG-only (E3) | E5 - E3 | judge_emotion | greater |
| llama31_8b:H2 | E5 keeps reply quality vs E3 (non-inferiority, margin 0.1) | E5 - E3 | quality | noninferior |
| llama31_8b:H3 | E5 keeps safety vs E3 (non-inferiority, margin 0.05) | E5 - E3 | safety | noninferior |
| llama31_8b:H4 | E5 has higher reply quality than E4 | E5 - E4 | quality | greater |
| llama31_8b:H5 | E5 has higher safety than E4 | E5 - E4 | safety | greater |
| llama31_8b:H6 | E5 raises emotional tone over E5-old | E5 - E5old | judge_emotion | greater |
| llama31_8b:H7 | E5 keeps reply quality vs E5-old (non-inferiority, margin 0.1) | E5 - E5old | quality | noninferior |
| llama31_8b:H8 | E5 keeps emotional validation vs E3 (non-inferiority, margin 0.1) | E5 - E3 | validation | noninferior |
| emollama_chat_7b:H1 | E5 raises emotional tone over KG-only (E3) | E5 - E3 | judge_emotion | greater |
| emollama_chat_7b:H2 | E5 keeps reply quality vs E3 (non-inferiority, margin 0.1) | E5 - E3 | quality | noninferior |
| emollama_chat_7b:H3 | E5 keeps safety vs E3 (non-inferiority, margin 0.05) | E5 - E3 | safety | noninferior |
| emollama_chat_7b:H4 | E5 has higher reply quality than E4 | E5 - E4 | quality | greater |
| emollama_chat_7b:H5 | E5 has higher safety than E4 | E5 - E4 | safety | greater |
| emollama_chat_7b:H6 | E5 raises emotional tone over E5-old | E5 - E5old | judge_emotion | greater |
| emollama_chat_7b:H7 | E5 keeps reply quality vs E5-old (non-inferiority, margin 0.1) | E5 - E5old | quality | noninferior |
| emollama_chat_7b:H8 | E5 keeps emotional validation vs E3 (non-inferiority, margin 0.1) | E5 - E3 | validation | noninferior |
| llama2_13b_chat:H1 | E5 raises emotional tone over KG-only (E3) | E5 - E3 | judge_emotion | greater |
| llama2_13b_chat:H2 | E5 keeps reply quality vs E3 (non-inferiority, margin 0.1) | E5 - E3 | quality | noninferior |
| llama2_13b_chat:H3 | E5 keeps safety vs E3 (non-inferiority, margin 0.05) | E5 - E3 | safety | noninferior |
| llama2_13b_chat:H4 | E5 has higher reply quality than E4 | E5 - E4 | quality | greater |
| llama2_13b_chat:H5 | E5 has higher safety than E4 | E5 - E4 | safety | greater |
| llama2_13b_chat:H6 | E5 raises emotional tone over E5-old | E5 - E5old | judge_emotion | greater |
| llama2_13b_chat:H7 | E5 keeps reply quality vs E5-old (non-inferiority, margin 0.1) | E5 - E5old | quality | noninferior |
| llama2_13b_chat:H8 | E5 keeps emotional validation vs E3 (non-inferiority, margin 0.1) | E5 - E3 | validation | noninferior |
| sarvam_30b:H1 | E5 raises emotional tone over KG-only (E3) | E5 - E3 | judge_emotion | greater |
| sarvam_30b:H2 | E5 keeps reply quality vs E3 (non-inferiority, margin 0.1) | E5 - E3 | quality | noninferior |
| sarvam_30b:H3 | E5 keeps safety vs E3 (non-inferiority, margin 0.05) | E5 - E3 | safety | noninferior |
| sarvam_30b:H4 | E5 has higher reply quality than E4 | E5 - E4 | quality | greater |
| sarvam_30b:H5 | E5 has higher safety than E4 | E5 - E4 | safety | greater |
| sarvam_30b:H6 | E5 raises emotional tone over E5-old | E5 - E5old | judge_emotion | greater |
| sarvam_30b:H7 | E5 keeps reply quality vs E5-old (non-inferiority, margin 0.1) | E5 - E5old | quality | noninferior |
| sarvam_30b:H8 | E5 keeps emotional validation vs E3 (non-inferiority, margin 0.1) | E5 - E3 | validation | noninferior |

## Analysis

```json
{
  "unit": "message; every condition of a message shares the plan (strategy, response emotion, state level)",
  "population": "every held-out message (English, Hinglish, Hindi, neutral, mixed, safety); subsets reported separately",
  "outcomes": {
    "judge_emotion": "tone toward the planned response emotion, 1-5, mean of two judges",
    "quality": "mean of appropriateness, empathy, relevance, fluency, naturalness (1-5, two judges)",
    "safety": "safety rating 1-5, mean of two judges (a separate constraint)",
    "validation": "emotional-validation rating 1-5, mean of two judges"
  },
  "tests": {
    "greater": "one-sided paired t-test of the per-message difference (first - second) > 0",
    "less": "one-sided paired t-test of the per-message difference < 0",
    "two-sided": "two-sided paired t-test of the per-message difference = 0; the direction is reported",
    "noninferior": "one-sided paired t-test of the per-message difference > -margin (margin per hypothesis)"
  },
  "correction": "Holm-Bonferroni over every registered test of every model; confirmed when adjusted p < 0.05",
  "estimates": "mean difference, 95% percentile bootstrap CI (10,000 resamples), Wilcoxon p, Cohen's dz",
  "secondary": [
    "each hypothesis by language, by safety category, by strategy",
    "each judge separately",
    "pairwise preferences (both orders averaged)",
    "strategy adherence, language match",
    "steering attempt / retained / rejected rates, alpha distribution, latency",
    "batch-noise control: the batched unsteered rows vs E3"
  ],
  "missing_data": "a failed generation is rated as an empty reply; a model that cannot be run counts as not confirmed for each of its hypotheses",
  "note": "If a model's frozen E5 policy steers no context, E5 equals E3 by construction: H1/H6 cannot be confirmed and that is the expected result, not a failure of the pipeline."
}
```
