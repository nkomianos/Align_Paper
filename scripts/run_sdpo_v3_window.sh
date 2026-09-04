#!/usr/bin/env bash
set -u
cd /home/ubuntu/sdpo_v3_source_20260904T0915Z
export PYTHONPATH="$PWD/src:$PWD"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
root=/home/ubuntu/sdpo_format_v3_20260904T0915Z
if [ -e "$root" ] || [ -e "$root.log" ]; then exit 73; fi
/home/ubuntu/Align_Paper/.venv-interaction/bin/python -m interaction_sprint.sdpo_format_positive_control_v3 /home/ubuntu/sdpo_v3_inputs_20260904T0915Z "$root" --model-path /home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --revision 1cfa9a7208912126459214e8b04321603b3df60c --upstream-file /home/ubuntu/sdpo_inputs/online_sdpo_updater.py --threads 4 --lr 0.0001 --seed 9047801 > "$root.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root.exit"
exit "$result"
