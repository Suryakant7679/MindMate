#!/usr/bin/env bash
# Every experiment for Llama-2-13B-chat, from recognition neurons to the dashboard evidence.
#
#   bash run_llama2_full.sh                 # all stages in order
#   bash run_llama2_full.sh battery single  # only the named stages (rerun a failed stage this way)
#
# Stages (each writes below results/llama2_13b_chat_*; logs in results/llama2_13b_chat_logs/):
#   recognition masking generation panels causal calibrate install heldout judge2 export
#                the per-model pipeline (run_model_pipeline.sh), one stage at a time
#   freeze       snapshot the installed panels, policy and evidence (needed by the two-judge studies)
#   twojudge     why contexts were rejected, two-judge consensus policy, second held-out comparison
#   v3           prompt ablation and steering-method comparison (calibration messages)
#   battery      unified causal validation battery (English + Hinglish), cross-model summary
#   single       single-neuron interventions
# Nothing is installed beyond the pipeline's own baseline policy; one model on the GPU at a time.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MINDMATE_MODEL=llama2
MODEL=llama2
LOGS=results/llama2_13b_chat_logs
mkdir -p "$LOGS"
CORE=(recognition masking generation panels causal calibrate install heldout judge2 export)
STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=("${CORE[@]}" freeze twojudge v3 battery single)

run() {      # run <stage> <command...>: log to file and terminal, stop the script if the command fails
    local stage="$1"; shift
    echo "===== $(date '+%F %T')  ${stage}: $* =====" | tee -a "$LOGS/${stage}.log"
    "$@" 2>&1 | tee -a "$LOGS/${stage}.log"
}

for STAGE in "${STAGES[@]}"; do
    case "$STAGE" in
    recognition|masking|generation|panels|causal|calibrate|install|heldout|judge2|export)
        run "$STAGE" bash run_model_pipeline.sh "$MODEL" "$STAGE" ;;
    freeze)
        if [[ -f results/frozen_v2/llama2_13b_chat/manifest.json ]]; then
            run freeze python3 -m generation_analysis.freeze_v2 --verify --models "$MODEL"
        else
            run freeze python3 -m generation_analysis.freeze_v2 --models "$MODEL"
        fi ;;
    twojudge)
        run twojudge python3 -m generation_analysis.diagnose_calibration
        run twojudge python3 -m generation_analysis.consensus_policy
        run twojudge python3 -m generation_analysis.holdout2 --stages generate score report ;;
    v3)
        run v3 python3 -m generation_analysis.v3_prompts --stages generate score report
        run v3 python3 -m generation_analysis.v3_methods --stages screen screen_score screen_report --prompt kg_v2
        run v3 python3 -m generation_analysis.v3_methods --stages calibrate calibrate_score select --prompt kg_v2 ;;
    battery)
        MINDMATE_VALIDATION_HINGLISH=1 run battery python3 -m generation_analysis.neuron_validation --model "$MODEL" all
        run battery python3 -m generation_analysis.neuron_validation summary ;;
    single)
        run single python3 -m generation_analysis.single_neuron --model "$MODEL" all ;;
    *)
        echo "Unknown stage: $STAGE" >&2; exit 2 ;;
    esac
    echo "===== $(date '+%F %T')  finished stage: ${STAGE} =====" | tee -a "$LOGS/progress.log"
done
echo "ALL REQUESTED LLAMA-2 STAGES FINISHED: ${STAGES[*]}" | tee -a "$LOGS/progress.log"
