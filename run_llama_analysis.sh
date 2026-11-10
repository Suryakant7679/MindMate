#!/usr/bin/env bash
# Reproduce the existing neuron-analysis sequence with Llama 3.1 8B Instruct.
# All outputs are isolated below results/llama31_8b_*.
set -euo pipefail
# Keeps long-running offloaded Llama evaluations from accumulating fragmented
# CUDA allocations between masking conditions.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ "${1:-}" == "--resume" ]]; then
    # The architecture report, full selectivity tensor, and layerwise top-1%
    # files already exist. Continue at the first lightweight failed stage.
    shift
elif [[ $# -eq 0 ]]; then
    python3 inspect_llama_architecture.py --load-weights
    python3 generate_full_selectivity.py
    python3 extract_layerwise_top1pct.py
else
    echo "Usage: bash run_llama_analysis.sh [--resume]" >&2
    exit 2
fi

python3 emotion_neuron_extraction.py --text-column hinglish_genz_text --label-column label --batch-size 4 \
    --selectivity-file results/llama31_8b_emotions_full/selectivity.npy
python3 analyze_emotion_association.py
python3 analyze_emotion_neuron_overlap.py
python3 analyze_layerwise_emotion_association.py
python3 05_train_emotion_probe.py --batch-size 4
python3 06_mask_single_emotion_neurons --batch-size 4
python3 07_mask_shared_emotion_neurons.py --batch-size 4
python3 08_random_matched_masking.py --batch-size 4
# Stages 09--11 use the exact-degree-1 neuron table derived from the
# layerwise top-1% extraction.  Generate it before launching those stages.
python3 analyze_top1pct_single_emotion.py
python3 09_mask_layerwise_top1pct_unique.py
python3 10_mask_all_top1pct_unique.py
python3 11_random_matched_all_top1pct_unique.py
python3 12_top5_selected_random_layerwise_overall.py
python3 13_select_single_joy_steering_neuron.py
python3 14_single_joy_neuron_steering.py
python3 15_direction_and_logit_analysis.py
python3 16_visualize_steering_direction.py
python3 17_rank_all_joy_actuator_neurons.py
python3 18_rank_all_top1pct_joy_actuators.py
python3 19_test_top_joy_actuators.py
python3 20_llama_recognition_generation_overlap.py
