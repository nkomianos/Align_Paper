#!/usr/bin/env bash
# Dependency-only retry. Scientific source remains the frozen T1030Z tree.
set -euo pipefail
SOURCE=/home/ubuntu/sdpo_single_profile_source_20260904T1030Z
RUN=/home/ubuntu/sdpo_single_profile_calibration_retry_20260904T1040Z
PYTHON=/home/ubuntu/sdpo_repro_env_20260904/bin/python
ASSETS=/home/ubuntu/sdpo_single_profile_assets_v2_20260904
UPSTREAM=/home/ubuntu/sdpo_single_profile_upstream_20260904
POLICY=/home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
SIMULATOR=/home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
POLICY_MANIFEST=/home/ubuntu/sdpo_format_v3_20260904T0915Z/model.json
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
export HF_HUB_CACHE=/home/ubuntu/Align_Paper/.hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false USE_TF=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
if [[ "${1:-}" == --worker ]]; then
    [[ ! -e "$RUN" ]] || exit 80
    set +e
    timeout --signal=TERM --kill-after=60s 7200s "$PYTHON" \
        "$SOURCE/src/interaction_sprint/sdpo_single_profile_reproduction.py" \
        calibration "$ASSETS" "$UPSTREAM" "$RUN" \
        --policy "$POLICY" --simulator "$SIMULATOR" --policy-manifest "$POLICY_MANIFEST"
    result=$?
    set -e
    printf '%s\n' "$result" > "$RUN.exit"
    exit "$result"
fi
[[ $# == 0 ]] || exit 81
for required in "$SOURCE/src/interaction_sprint/sdpo_single_profile_reproduction.py" "$ASSETS/data_manifest.json" \
    "$ASSETS/model_manifest.json" "$UPSTREAM/PINNED_SOURCE.json" "$POLICY_MANIFEST" "$PYTHON"; do
    [[ -f "$required" ]] || { echo "Missing staged input: $required" >&2; exit 82; }
done
for reserved in "$RUN" "$RUN.log" "$RUN.pid" "$RUN.exit" "$RUN.launch-lock"; do
    [[ ! -e "$reserved" ]] || { echo "Refusing to overwrite: $reserved" >&2; exit 83; }
done
mkdir "$RUN.launch-lock"
SELF=$(realpath -- "$0")
nohup bash "$SELF" --worker > "$RUN.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN.pid"
printf 'Calibration-only dependency retry launched; worker PID %s\n' "$(< "$RUN.pid")"
