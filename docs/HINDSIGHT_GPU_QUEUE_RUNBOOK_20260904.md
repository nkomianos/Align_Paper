# Hindsight decisive GPU queue

> **Audit hold — 5 September 2026 UTC.** This runbook is preserved as a historical
> frozen queue, not current launch authorization. The
> [independent audit](ICLR_2027_INDEPENDENT_AUDIT_REPORT_20260905.md) found endpoint,
> baseline, statistical-unit, verifier and estimand defects. Do not launch this
> sequence unchanged. A proposed reduced study and conditional queue are in the
> [paper completion plan](ICLR_2027_PAPER_COMPLETION_PLAN_20260905.md); they require
> their own prospective implementation/validation/freeze before any model endpoint.

## Frozen transfer inputs

- Source bundle and checksum are refreshed after the policy-G1 implementation
  commit. Use the latest non-overwriting `hindsight_gpu_queue_*.bundle` and its
  adjacent receipt rather than the superseded `e522ad6` bundle.
- The exact detached commit used on the host must equal the branch tip recorded
  by `git bundle list-heads` and the commit named in every launcher variable.
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
5. If and only if gradient G0 v2 qualifies and verifies, run the frozen neural
   policy-learning G1 using that exact G0 root as its prerequisite. Preserve all
   26 trained arms and verify its evidence.
6. Run the independent PUPPET capable-reader DEV regardless of the gradient's
   scientific decision, provided the runtime and private-input checks pass.
7. Secure and verify that evidence. Do not run the locked human confirmation
   split or any post-G1 expansion automatically.
8. If and only if G1 qualifies, stage and verify the full-learning EndoPAHF v3
   input root, then run exact-interface preflight v2 against that root. The
   superseded extraction-query preflight must not be used.
9. If and only if that preflight qualifies, run repaired G2 v2 DEV. It trains
   15 frozen arms for 35 steps across four panels, covers every one of the 630
   learning bases once per arm, and never reads EndoPAHF confirmation.
10. Secure and independently verify G2 DEV with the actual G1, preflight and
    input roots. Only a qualified DEV may authorize the separate locked
    EndoPAHF confirmation runner. That runner evaluates frozen adapters and
    performs no training or model selection.

Estimated paid time after the model is cached is 25--70 minutes for gradient G0,
4--8 hours for conditional policy G1, and 30--60 minutes for the capable reader.
Allow another 15--40 minutes on a fresh host for environment validation and
model download. The pass-path total is approximately 5--10.2 hours; a failed G0
skips G1 and reduces the total to about 1.2--2.8 hours.

If G1 passes, budget roughly 4--8 additional hours for exact-interface
preflight, G2 DEV and conditional confirmation. The complete synthetic-plus-
external pass path is therefore approximately 9--18 hours, but every expensive
stage is skipped immediately when its prerequisite fails.

## Remote launch contract

All launchers require an absolute fresh output root, exact clean Git commit,
one CUDA device, Transformers with native `Qwen3_5ForCausalLM`, and an explicit
Python runtime. The human launcher also verifies the private dataset SHA-256
before model loading. Policy G1 additionally reruns the committed G0 verifier
inside its runner and records the prerequisite manifest and result hashes. All
three launchers write a remote checksum for the sealed manifest.

Gradient environment variables:

```bash
export HINDSIGHT_GRADIENT_V2_ROOT=/home/ubuntu/hindsight_gradient_g0_v2_TIMESTAMP
export HINDSIGHT_GRADIENT_V2_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
export HINDSIGHT_GRADIENT_V2_PINNED_COMMIT=EXACT_TRANSFER_COMMIT
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
bash scripts/run_hindsight_neural_gradient_g0_v2_remote.sh
```

Conditional policy-G1 environment variables:

```bash
export HINDSIGHT_POLICY_G1_ROOT=/home/ubuntu/hindsight_policy_g1_TIMESTAMP
export HINDSIGHT_POLICY_G1_GRADIENT_ROOT=/home/ubuntu/hindsight_gradient_g0_v2_TIMESTAMP
export HINDSIGHT_POLICY_G1_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
export HINDSIGHT_POLICY_G1_PINNED_COMMIT=EXACT_TRANSFER_COMMIT
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
bash scripts/run_hindsight_neural_policy_g1_remote.sh
```

Human-reader environment variables:

```bash
export HINDSIGHT_HUMAN_LLM_ROOT=/home/ubuntu/hindsight_human_llm_dev_TIMESTAMP
export HINDSIGHT_HUMAN_LLM_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
export HINDSIGHT_HUMAN_LLM_PINNED_COMMIT=EXACT_TRANSFER_COMMIT
export PUPPET_DATA=/home/ubuntu/frozen_inputs/puppet/hidden_puppet_master_dataset.csv
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
bash scripts/run_hindsight_human_feedback_llm_dev_remote.sh
```

Conditional EndoPAHF v3 preflight variables:

```bash
export HINDSIGHT_PAHF_V3_INPUT_ROOT=/home/ubuntu/frozen_inputs/hindsight_endo_pahf_v3
export HINDSIGHT_PAHF_PREFLIGHT_ROOT=/home/ubuntu/hindsight_pahf_preflight_TIMESTAMP
export HINDSIGHT_PAHF_PREFLIGHT_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
export HINDSIGHT_PAHF_PREFLIGHT_PINNED_COMMIT=EXACT_TRANSFER_COMMIT
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
bash scripts/run_hindsight_pahf_preflight_remote.sh
```

Conditional EndoPAHF G2 v2 DEV variables:

```bash
export HINDSIGHT_PAHF_V3_INPUT_ROOT=/home/ubuntu/frozen_inputs/hindsight_endo_pahf_v3
export HINDSIGHT_PAHF_G2_G1_ROOT=/home/ubuntu/hindsight_policy_g1_TIMESTAMP
export HINDSIGHT_PAHF_G2_PREFLIGHT_ROOT=/home/ubuntu/hindsight_pahf_preflight_TIMESTAMP
export HINDSIGHT_PAHF_G2_DEV_ROOT=/home/ubuntu/hindsight_pahf_g2_v2_dev_TIMESTAMP
export HINDSIGHT_PAHF_G2_PYTHON=/home/ubuntu/Align_Paper/.venv/bin/python
export HINDSIGHT_PAHF_G2_PINNED_COMMIT=EXACT_TRANSFER_COMMIT
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
bash scripts/run_hindsight_pahf_g2_dev_remote.sh
```

Do not set `HINDSIGHT_PAHF_G2_CONFIRM_ROOT` or run the confirmation launcher
unless the DEV root has first passed the committed verifier. Confirmation uses
the same variables plus a fresh confirmation root.

When launching non-interactively, redirect each command to its own log and save
its PID beside the output root. Never reuse a root after any failure.

## Retrieval

After each process exits, archive its complete root plus adjacent manifest hash,
log, PID and exit receipt. Compute the archive SHA-256 remotely, copy it into a
fresh non-overwriting directory under `retrieved/`, verify the archive hash, and
run the corresponding committed read-only verifier. Do not delete the remote
copy until local verification succeeds and the user explicitly terminates the
instance.
