#!/usr/bin/env bash
# Fills the dashboard's missing evidence: Sarvam per-neuron gradient estimates, Hinglish battery for the Llama family.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -m generation_analysis.neuron_validation --model sarvam gradients analyse || echo "SARVAM GRADIENTS FAILED"
for model in llama emollama; do
  MINDMATE_VALIDATION_HINGLISH=1 python3 -m generation_analysis.neuron_validation --model $model generate score analyse || echo "$model HINGLISH FAILED"
done
python3 -m generation_analysis.neuron_validation summary
echo "ALL GAP JOBS FINISHED"
