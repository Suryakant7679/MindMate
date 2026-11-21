#!/usr/bin/env bash
# Live GPU evaluation of the Safety & Risk Assessment Agent (Agent 1).  Every stage logs to the terminal and to
# results/safety_eval/logs/<stage>.log; the script stops at the first failure.
#
#   bash run_safety_eval.sh                       # check smoke llama sarvam report regression
#   bash run_safety_eval.sh llama report          # only the named stages
#
# Stages
#   check        packages, GPU, models on disk (no model loaded)
#   regression   all existing unit tests (no GPU)
#   baselines    model-free systems A, B (rules), C without a model, on dev + heldout (no GPU)
#   smoke        Llama: 3 cases per dataset through every part (minutes; catches setup problems)
#   llama        Llama 3.1-8B: classify (C, B-LLM, no-LLM) + outgoing + HTTP pipeline (chat + stream + injection)
#   sarvam       Sarvam-30B (4-bit): the same
#   ablations    Llama: the remaining ablations of C (no_prior, no_screen, no_normalization); reuses cached calls
#   report       combine all runs into results/safety_eval/report/REPORT.md and CSV tables
# Environment overrides: MODEL_ID_LLAMA / MODEL_ID_SARVAM (local checkpoint folders), EXTRA_ARGS (passed to live_eval).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=${PYTHON:-python3}
LOGS=results/safety_eval/logs
mkdir -p "$LOGS"
STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(check smoke llama sarvam report regression)

run() {      # run <stage> <command...>
    local stage="$1"; shift
    echo "===== $(date '+%F %T')  ${stage}: $* =====" | tee -a "$LOGS/${stage}.log"
    "$@" 2>&1 | tee -a "$LOGS/${stage}.log"
}
model_id() {  # optional local checkpoint folder for a preset
    local value="$1"; [[ -n "$value" ]] && echo "--model-id $value"
}

for STAGE in "${STAGES[@]}"; do
    case "$STAGE" in
        check)
            run check "$PY" -m chatbot_feature.evaluation.safety.check_env ;;
        regression)
            run regression "$PY" -m unittest discover -s chatbot_feature/tests -t .
            run regression "$PY" -m unittest discover -s generation_analysis/tests -t .
            run regression "$PY" -m unittest tests.test_chat tests.test_emotion_service tests.test_research \
                tests.test_validation_battery tests.test_models tests.test_sarvam_steering tests.test_workspace_api ;;
        baselines)
            run baselines "$PY" -m chatbot_feature.evaluation.safety.live_eval --no-model ;;
        smoke)
            # shellcheck disable=SC2046
            run smoke "$PY" -m chatbot_feature.evaluation.safety.live_eval --model llama $(model_id "${MODEL_ID_LLAMA:-}") \
                --limit 3 --inject 1 --out results/safety_eval/smoke_llama ${EXTRA_ARGS:-} ;;
        llama)
            # shellcheck disable=SC2046
            run llama "$PY" -m chatbot_feature.evaluation.safety.live_eval --model llama $(model_id "${MODEL_ID_LLAMA:-}") \
                ${EXTRA_ARGS:-} ;;
        sarvam)
            # shellcheck disable=SC2046
            run sarvam "$PY" -m chatbot_feature.evaluation.safety.live_eval --model sarvam $(model_id "${MODEL_ID_SARVAM:-}") \
                ${EXTRA_ARGS:-} ;;
        ablations)
            # shellcheck disable=SC2046
            run ablations "$PY" -m chatbot_feature.evaluation.safety.live_eval --model llama $(model_id "${MODEL_ID_LLAMA:-}") \
                --parts classify --ablations full context_free no_prior no_screen no_normalization no_llm ${EXTRA_ARGS:-} ;;
        report)
            run report "$PY" -m chatbot_feature.evaluation.safety.report ;;
        *)
            echo "Unknown stage: $STAGE (check regression baselines smoke llama sarvam ablations report)"; exit 2 ;;
    esac
    echo "===== $(date '+%F %T')  finished stage: $STAGE =====" | tee -a "$LOGS/progress.log"
done
echo "SAFETY EVALUATION STAGES FINISHED: ${STAGES[*]}" | tee -a "$LOGS/progress.log"
