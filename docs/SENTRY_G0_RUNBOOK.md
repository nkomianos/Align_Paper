# SENTRY G0 execution and interpretation

## What this run tests

G0 is an end-to-end **forecasting calibration** test. For sixteen precommitted
benign post-training batches (eight calibration, eight sealed), it asks whether
four fixed rank-4 shadows predict the signed drift of two rank-32 Qwen3.5-9B
LoRA runs. Each batch has a matched source prompt count and target length.
Training examples intentionally carry no scenario ID, partition membership, or
source identifier.

The two channels are a balanced vocabulary response choice and a safe response
body-organization choice. This is not a covert-transfer reproduction and not a
safety evaluation. Its sole value is to falsify a proposed audit before a more
expensive real transfer replication.

## Frozen sequence

1. Stage exact public repositories/datasets and extract question-only JSONL.
2. Attest the model/runtime/public contract without reading the key.
3. Create a private membership/direction commitment outside the run root.
4. Train two full and four shadow adapters per batch, preserving every adapter
   and JSONL input.
5. Recompute the fail-closed gate from raw scenario evidence and write a hash
   bound report. Retrieval uses the offline verifier; it does not need the key.

The implementation refuses existing destinations at each irreversible step.
Interrupted output must be retained and a new output root used for another run.

## Operator invocation

On the GH200 after the repository and its GPU environment are ready:

```bash
export SENTRY_PUBLIC_SOURCE_ROOT=/mnt/sentry/public-YYYYMMDD
export SENTRY_PREFLIGHT_ROOT=/mnt/sentry/preflight-YYYYMMDD
export SENTRY_RUNTIME_PREFLIGHT_ROOT=/mnt/sentry/runtime-preflight-YYYYMMDD
export SENTRY_G0_OUTPUT_ROOT=/mnt/sentry/g0-YYYYMMDD
export SENTRY_PRIVATE_KEY_ROOT=/mnt/sentry-private/g0-key-YYYYMMDD
./scripts/stage_sentry_public_sources_remote.sh
./scripts/run_sentry_g0_remote.sh
```

All four paths must be fresh. The private key path must not be under the run
root. The first invocation intentionally stops after its runtime preflight.
After inspecting its measured throughput/memory report, use fresh paths and
`SENTRY_SKIP_RUNTIME_PREFLIGHT=1` for the full gate. The model and datasets are
public; a Hugging Face token is only needed if the remote environment cannot
access public Hub resources.

## Decision rule

Only `PROCEED_TO_EXTERNAL_REPLICATION` is a G0 pass. It requires the frozen
rank correlation, recall/FPR, AUC margin against all four baselines, both
channel controls, and shadow compute budget. Any other decision is
`KILL_SHADOW_STUDENT_CANDIDATE`; no threshold tuning or deletion of a channel
is permitted. Even a pass only authorizes the teacher-generated covert-transfer
replication and independent-family validation, not a paper claim.

## Expected cost

The frozen workload contains 32 full 1.92M-token adapters and 64 shadow
240k-token adapters, plus base/shadow scoring. On the GH200 this needs an
empirical speed preflight before estimating wall time. The previous 4–8 GPU
hour estimate is now an optimistic lower bound, not a commitment: exact Qwen
DeltaNet backward throughput and adapter-save bandwidth will determine the
real cost. A first preflight should measure one full and one shadow endpoint;
do not infer cost from an unmeasured token rate.
