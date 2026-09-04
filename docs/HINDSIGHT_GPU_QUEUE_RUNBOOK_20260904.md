# Hindsight decisive GPU queue

## Frozen transfer inputs

- Source bundle:
  `artifacts/deployment/hindsight_gpu_queue_e522ad6.bundle`
- Bundle SHA-256:
  `93b0098eacc7d6cd0161c6e59372baa5814d631b5a4f9b04834caadf812bc12b`
- Bundle branch tip and required detached commit:
  `e522ad62c2a5fa812f55e252140c922f178e8485`
- Private PUPPET input:
  `artifacts/puppet_schema_20260904_v1/hidden_puppet_master_dataset.csv`
- PUPPET SHA-256:
  `6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d`

The bundle was independently verified by `git bundle verify` and records the
complete branch history. The private CSV is ignored by Git and must never be
committed or placed in a public archive.

## Order and stopping rules

1. Restore the bundle into a clean detached checkout at the exact commit above.
2. Use a pre-existing verified `.venv` and model cache only if their receipts
   match the host; otherwise run `scripts/bootstrap_lambda.sh` once.
3. Run neural-gradient G0 v2. Its semantic-interface qualification can stop the
   assay before any gradient. If it qualifies, preserve all 48 gradient vectors.
4. Secure and verify that evidence.
5. Run the independent PUPPET capable-reader DEV regardless of the gradient's
   scientific decision, provided the runtime and private-input checks pass.
6. Secure and verify that evidence. Do not run the locked human confirmation
   split or any policy-learning expansion automatically.

Estimated paid time after the model is cached is 25--70 minutes for gradient G0
and 30--60 minutes for the capable reader. Allow another 15--40 minutes on a
fresh host for environment validation and model download. The expected total is
approximately 1.2--2.8 hours, with earlier fail-closed stops possible.

## Remote launch contract

Both launchers require an absolute fresh output root, exact clean Git commit,
one CUDA device, Transformers with native `Qwen3_5ForCausalLM`, and an explicit
Python runtime. The human launcher also verifies the private dataset SHA-256
before model loading. They write a remote checksum for the sealed manifest.

Gradient environment variables:

```bash
export HINDSIGHT_GRADIENT_V2_ROOT=/home/ubuntu/hindsight_gradient_g0_v2_TIMESTAMP
export HINDSIGHT_GRADIENT_V2_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
export HINDSIGHT_GRADIENT_V2_PINNED_COMMIT=e522ad62c2a5fa812f55e252140c922f178e8485
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
bash scripts/run_hindsight_neural_gradient_g0_v2_remote.sh
```

Human-reader environment variables:

```bash
export HINDSIGHT_HUMAN_LLM_ROOT=/home/ubuntu/hindsight_human_llm_dev_TIMESTAMP
export HINDSIGHT_HUMAN_LLM_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
export HINDSIGHT_HUMAN_LLM_PINNED_COMMIT=e522ad62c2a5fa812f55e252140c922f178e8485
export PUPPET_DATA=/home/ubuntu/frozen_inputs/puppet/hidden_puppet_master_dataset.csv
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
bash scripts/run_hindsight_human_feedback_llm_dev_remote.sh
```

When launching non-interactively, redirect each command to its own log and save
its PID beside the output root. Never reuse a root after any failure.

## Retrieval

After each process exits, archive its complete root plus adjacent manifest hash,
log, PID and exit receipt. Compute the archive SHA-256 remotely, copy it into a
fresh non-overwriting directory under `retrieved/`, verify the archive hash, and
run the corresponding committed read-only verifier. Do not delete the remote
copy until local verification succeeds and the user explicitly terminates the
instance.
