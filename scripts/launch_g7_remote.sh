#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/align_research_20260910
source env/bin/activate
exec python scripts/run_g7_queue.py \
  --config configs/memory_graft_security_g7_preregistered.json \
  --preregistration docs/MEMORY_GRAFT_SECURITY_G7_PREREGISTRATION_20260913.md \
  --receipt configs/memory_graft_security_g7_freeze_receipt.json \
  --data g7_fineweb_1b_v1 \
  --compression memory_graft_security_s1_v1_1_run1/compression.npy \
  --wikitext-root memory_graft_security_s1_v1_1_run1 \
  --model-cache model_cache \
  --output memory_graft_security_g7_run1
