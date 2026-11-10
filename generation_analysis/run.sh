#!/usr/bin/env bash
# Generation-time emotion-neuron analysis for the original Llama 3.1 8B Instruct
# checkpoint (no LoRA).  Reads recognition results; writes only below
# results/llama31_8b_generation_emotion.  Run from the project root.
# Pass --resume to continue interrupted generation/steering stages.
set -euo pipefail
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
RESUME="${1:-}"

python3 -m generation_analysis.capture --batch-size 8 ${RESUME}
python3 -m generation_analysis.score_tokens --batch-size 8
python3 -m generation_analysis.identify_neurons
python3 -m generation_analysis.steering --batch-size 8 ${RESUME}
