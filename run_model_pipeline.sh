#!/usr/bin/env bash
# Run the whole analysis sequence for one chat model, from recognition neurons to the
# held-out steering comparison -- the same steps that were run for Llama 3.1-8B.
#
#   bash run_model_pipeline.sh <model> [stage ...]
#
#   <model>  llama | llama2 | emollama | sarvam      (keys of backend/config.py MODEL_REGISTRY)
#   stages   recognition  masking  generation  panels  causal  calibrate  install  heldout  judge2  export
#            (default: all, in this order).  Every stage writes below results/<tag>_*, so models
#            never overwrite each other, and a failed stage can be rerun on its own.
#
# The model must already be downloaded (`hf download <hugging-face-id>`); nothing is downloaded here
# unless MINDMATE_ALLOW_DOWNLOAD=1.  Run from the project root, one model at a time (one GPU).
set -euo pipefail
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ $# -lt 1 ]]; then
    sed -n '2,13p' "$0" >&2
    exit 2
fi
export MINDMATE_MODEL="$1"
shift
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(recognition masking generation panels causal calibrate install heldout judge2 export)
fi
if [[ "${MINDMATE_ALLOW_DOWNLOAD:-0}" != "1" ]]; then
    export HF_HUB_OFFLINE=1
fi

# Model id, result prefix and "is it on disk" come from the same registry the chat backend uses.
python3 -c "import experiment_model as m; m.require_downloaded()"
TAG="$(python3 -c 'import experiment_model as m; print(m.EXPERIMENT_TAG)')"
GEN="results/${TAG}_generation_emotion"
echo "Model: ${MINDMATE_MODEL}   results: results/${TAG}_*   stages: ${STAGES[*]}"

for STAGE in "${STAGES[@]}"; do
    echo "===== ${MINDMATE_MODEL}: ${STAGE} ====="
    case "${STAGE}" in
    recognition)
        # Recognition neurons (dataset-text forward passes), top-1% tables and the frozen probe.
        python3 inspect_llama_architecture.py --load-weights
        python3 generate_full_selectivity.py
        python3 extract_layerwise_top1pct.py
        python3 emotion_neuron_extraction.py --text-column hinglish_genz_text --label-column label --batch-size 4 \
            --selectivity-file "results/${TAG}_emotions_full/selectivity.npy"
        python3 analyze_emotion_association.py
        python3 analyze_emotion_neuron_overlap.py
        python3 analyze_layerwise_emotion_association.py
        python3 05_train_emotion_probe.py --batch-size 4
        ;;
    masking)
        # Masking and single-neuron studies of the recognition neurons.
        python3 06_mask_single_emotion_neurons --batch-size 4
        python3 07_mask_shared_emotion_neurons.py --batch-size 4
        python3 08_random_matched_masking.py --batch-size 4
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
        ;;
    generation)
        # Generation-time neurons: matched responses, token scoring, neuron sets, selectivity steering.
        python3 -m generation_analysis.capture --batch-size 8
        python3 -m generation_analysis.score_tokens --batch-size 8
        python3 -m generation_analysis.identify_neurons
        python3 -m generation_analysis.steering --batch-size 8
        python3 -m generation_analysis.causal_validation
        ;;
    panels)
        # Attribution-selected steering panels (layer band scales with the model's depth) and a first check.
        python3 -m generation_analysis.build_steering_panels
        python3 -m generation_analysis.evaluate_chat_steering --targets auto mirror joy admiration
        ;;
    causal)
        # Frozen dev / calibration / test splits, then the causal panel experiment on the dev split.
        python3 -c "from generation_analysis.steering_lab import build_splits; build_splits()"
        python3 -m generation_analysis.causal_panel_experiment
        ;;
    calibrate)
        # Steering strengths on the calibration split for the attribution panel and two causal candidates.
        python3 -m generation_analysis.recalibrate_policy \
            --panels deployed cand:deployed_causal_half cand:grad_global_half
        ;;
    install)
        # Baseline choice: the attribution panel (panels.json) with its own recalibrated policy.
        # To use a causal candidate instead, copy its policy_cand_*.json here and name its panel file
        # in the model's ModelConfig.panel_file (backend/config.py) -- only after the calibration
        # tables support it, as was done for Llama 3.1-8B.
        mkdir -p "${GEN}/steering_policy"
        if [[ -f "${GEN}/steering_policy/policy.json" ]]; then
            cp "${GEN}/steering_policy/policy.json" "${GEN}/steering_policy/policy_before_install.json"
        fi
        cp "${GEN}/causal_steering/recalibration/policy_deployed.json" "${GEN}/steering_policy/policy.json"
        echo "Installed ${GEN}/steering_policy/policy.json"
        ;;
    heldout)
        # Test split, read once: MindMate prompt only vs strategy prompt vs the installed system.
        # (The frozen previous system, stage v1, exists only for Llama 3.1-8B.)
        if [[ "${MINDMATE_MODEL}" == "llama" ]]; then
            python3 -m generation_analysis.heldout_comparison --stages v1 v2 score report
        else
            python3 -m generation_analysis.heldout_comparison --stages v2 score report
        fi
        ;;
    judge2)
        # Second judge (Gemma 3-27B, 4-bit) on the held-out replies and the calibration sweep.
        python3 -m generation_analysis.second_judge --load-in-4bit --panel deployed
        ;;
    export)
        python3 -m generation_analysis.export_steering_evidence --variant deployed
        ;;
    *)
        echo "Unknown stage: ${STAGE}" >&2
        exit 2
        ;;
    esac
done
echo "Done: ${MINDMATE_MODEL} (${STAGES[*]})"
