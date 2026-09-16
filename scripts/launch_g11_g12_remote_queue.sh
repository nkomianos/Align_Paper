#!/usr/bin/env bash
set -u
cd /home/ubuntu/align_research_20260910
while kill -0 829391 2>/dev/null; do sleep 60; done
if [[ -f memory_graft_security_g11_run1/COMPLETE ]]; then
  env/bin/python scripts/run_memory_graft_security_g12.py --config configs/memory_graft_security_g12_preregistered.json --preregistration docs/MEMORY_GRAFT_SECURITY_G12_PREREGISTRATION_20260915.md --receipt configs/memory_graft_security_g12_freeze_receipt.json --s1-root memory_graft_security_s1_v1_1_run1 --model-cache model_cache --output memory_graft_security_g12_run1 > g12_source.log 2>&1
  g12_status=$?
  env/bin/python scripts/run_memory_graft_security_g11.py --config configs/memory_graft_security_g11_preregistered.json --preregistration docs/MEMORY_GRAFT_SECURITY_G11_PREREGISTRATION_20260915.md --receipt configs/memory_graft_security_g11_freeze_receipt.json --s1-root memory_graft_security_s1_v1_1_run1 --model-cache model_cache --output memory_graft_security_g11_replay > g11_replay.log 2>&1
  if [[ $g12_status -eq 0 && -f memory_graft_security_g12_run1/COMPLETE ]]; then
    env/bin/python scripts/run_memory_graft_security_g12.py --config configs/memory_graft_security_g12_preregistered.json --preregistration docs/MEMORY_GRAFT_SECURITY_G12_PREREGISTRATION_20260915.md --receipt configs/memory_graft_security_g12_freeze_receipt.json --s1-root memory_graft_security_s1_v1_1_run1 --model-cache model_cache --output memory_graft_security_g12_replay > g12_replay.log 2>&1
  fi
else
  echo "G11 source did not complete; queue stopped before G12" > g11_g12_queue_failure.log
fi