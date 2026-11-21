#!/usr/bin/env bash
# Resume research only. Installation requires a separate evidence review.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec 9>results/.sarvam_research.lock
flock -n 9 || { echo 'Another research runner is active.' >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
lab=results/sarvam_30b_generation_emotion/steering
stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive="results/sarvam_takeover/$stamp"
mkdir -p "$archive"
cp -a results/sarvam_30b_generation_emotion "$archive/"
cp -a generation_analysis/sarvam_experiments.py generation_analysis/sarvam_lab.py "$archive/"
exec > >(tee -a "$archive/run.log") 2>&1
echo "Snapshot and log: $archive"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PYTHON_BIN" -m unittest tests.test_sarvam_steering tests.test_emotion_service tests.test_chat tests.test_models tests.test_workspace_api
args=(--alphas 1 2 3 4 --language-alphas 0.5 0.75 1.0 1.25 --combo-language 0.75 --max-new-tokens 100)
if [[ ! -f "$lab/heldout/lock.json" ]]; then
    "$PYTHON_BIN" -m generation_analysis.sarvam_experiments causal3 causal3_score causal3_report "${args[@]}"
    "$PYTHON_BIN" -m generation_analysis.sarvam_experiments calibrate calibrate_score compare select "${args[@]}"
fi
"$PYTHON_BIN" -m generation_analysis.sarvam_experiments heldout heldout_score heldout_report evidence "${args[@]}"
echo 'Research stages finished. NOT INSTALLED. Real-model chat/LangGraph tests and independent benefit/quality review remain required.'
