# Lambda H100 runbook: bridge first

This runbook targets one x86_64 NVIDIA H100 PCIe 80 GB instance. The first paid experiment is the paired, same-environment G-RL/P-RL bridge. The older twelve-adapter symbolic SFT matrix is apparatus calibration, not the primary paid result.

## Decision boundary

The bridge starts two policies from paired copies of the same pinned base model. Both encounter the same worlds, actions, order, opportunity count, and optimization budget. G-RL receives genuine-outcome reward; P-RL receives proxy/evaluator reward. Formal Stage 1 parses and evaluates development records only. Locked-test bytes are transferred and hash-verified as part of the frozen archive, but the program does not parse those records until a human reviews Stage 1, the Stage 1 gate passes, and an exact approval file is created.

No script automatically launches replication. A bridge pass establishes only that the assay distinguishes these learned reward-channel controllers under the preregistered interventions. It does not establish intrinsic goals, general alignment, or future-hack prediction.

## Before launch

Inputs required from the operator:

- SSH host/user/key access to the H100 instance;
- the instance ID, displayed hourly price, immutable image ID when available, and
  launch time;
- a hard dollar/time ceiling (the frozen pilot assumes at most $200 before tax);
- a Lambda API key kept only on the local machine for the independent termination
  watchdog;
- outbound HTTPS access from the instance to PyPI and Hugging Face.

Do not send or commit private keys, API keys, or access tokens. A Hugging Face
token is optional and, if rate limiting makes one necessary, should be read-only.
No paid model API account or dataset credential is required.

On the local machine:

```powershell
cd <path-to-Align_Paper>
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m pytest tests -q
python -m under_extinction --config configs/bridge_smoke.yaml bridge-build
python -m under_extinction --config configs/bridge_smoke.yaml bridge-oracle --split dev
python -m under_extinction --config configs/bridge_pilot.yaml bridge-build
python -m under_extinction --config configs/bridge_pilot.yaml bundle --destination deployment/extinction_bridge_qwen35_9b_pilot.tar.gz
Get-FileHash deployment/extinction_bridge_qwen35_9b_pilot.tar.gz -Algorithm SHA256
```

Do not launch unless tests pass, both bridge builds validate their hashes, the oracle is explicitly labeled non-empirical, the model is exactly `Qwen/Qwen3.5-9B` at frozen revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, the text-only/non-thinking architecture contract is present, and the bundle checksum matches its `.sha256` file.

Confirm in the Lambda console:

- one H100 PCIe 80 GB, x86_64;
- the Lambda Stack 22.04 x86-64 image and, if exposed, its immutable image ID;
- the displayed hourly rate and applicable tax;
- at least 150 GiB free local disk for the frozen bundle, model cache, adapters,
  optimizer checkpoints, predictions, and retrieval archive;
- working SSH access;
- a local machine that can stay awake with the independent termination watchdog armed.

At the rate recorded on 2026-08-16, $3.29/hour, a 15% reserve leaves 25.84 hours from $100 or 51.67 hours from $200. Treat the console price as authoritative. Compute the termination deadline from instance launch/health, not from training start.

## Transfer, verify, and bootstrap

**Do not begin paid setup yet:** immediately after the instance becomes healthy,
first complete the next two sections (“Establish the absolute cost boundary” and
“Arm the independent termination watchdog”).  Confirm the armed watchdog in its
own local terminal, then return here.  Bootstrap is covered by that external hard
deadline even though it is not itself an experiment command.

Copy both the archive and checksum. On the GPU, verify before unpacking:

```bash
sha256sum --check extinction_bridge_qwen35_9b_pilot.tar.gz.sha256
tar -xzf extinction_bridge_qwen35_9b_pilot.tar.gz
cd under_extinction
set -o pipefail
bash scripts/bootstrap_lambda.sh 2>&1 | tee bootstrap_lambda.log
python -m under_extinction --config configs/bridge_pilot.yaml install-data --source frozen_data
```

Bootstrap validates an H100 with at least 75 GiB, Python >=3.10,<3.14, and the
preinstalled CUDA PyTorch >=2.5,<3. It creates a virtual environment with system
site packages, installs the pinned released Transformers/PEFT stack, verifies the
Qwen3.5 text loader is importable, and runs CPU tests. The following `install-data`
command verifies the exact transferred formal corpus against its manifest and
installs it at the configured output path; Stage 1 must reuse that corpus.
Bootstrap does not change drivers or compile source-only acceleration kernels. If
CUDA or the released software contract is broken, retrieve the bootstrap log and
terminate instead of repairing the paid image.

The frozen first-run kernel policy is `torch_fallback_required`. Qwen3.5 can use
optional causal-convolution and FLA CUDA extensions, but the causal-convolution
package has no ordinary pinned binary wheel for this stack and Transformers 5.15
can route through dynamically fetched kernels. The experiment explicitly passes
`use_kernels=False`, forbids those optional backends, and requires the released
Transformers PyTorch implementation. The exact-model throughput projection is
therefore the fail-closed cost gate. A later kernel-optimized replication would
be a separately frozen runtime, not an in-place repair or unreported deviation.

No Hugging Face token is normally required: the pinned Qwen repository is public.
If Hugging Face rate limiting requires one, use a read-only token, export it
interactively as `HF_TOKEN`, never put it in a command argument or file, and verify
that it is absent from manifests and logs. No OpenAI, Anthropic, or other model API
key is used by this experiment.

## Establish the absolute cost boundary

Choose one UTC termination time. Use the same instant for the local watchdog and `UE_HARD_DEADLINE_EPOCH` on the GPU. Example placeholders:

```bash
export UE_INSTANCE_TYPE='gpu_1x_h100_pcie'
export UE_INSTANCE_ID='INSTANCE_ID_FROM_LAUNCH_DETAILS'
export UE_HOURLY_USD='3.29'
export UE_LAMBDA_IMAGE_ID='IMAGE_ID_FROM_LAUNCH_DETAILS'
export UE_INSTANCE_LAUNCHED_AT='YYYY-MM-DDTHH:MM:SSZ'
UE_TERMINATE_AT_UTC='YYYY-MM-DDTHH:MM:SSZ'
export UE_HARD_DEADLINE_EPOCH="$(date --date="$UE_TERMINATE_AT_UTC" +%s)"
test "$UE_HARD_DEADLINE_EPOCH" -gt "$(date +%s)"
```

Each paid bridge script parses `UE_INSTANCE_LAUNCHED_AT` into an exported
`UE_INSTANCE_START_EPOCH`, records both values, and rejects a missing or future
launch time. It also records the exact `UE_INSTANCE_ID` and rejects a missing or
malformed value, a nonnumeric `UE_HOURLY_USD`, or any hourly price that
differs from the frozen config. If the console price has changed, update and
refreeze the config and data locally rather than editing it on the paid instance.
Stage 1 requires the same instance ID and launch epoch attested by preflight and
rejects any later extension of the preflight termination deadline.

Every bridge script rejects a missing or nonnumeric deadline. It subtracts `budget.retrieval_reserve_minutes` from that deadline before running paid compute and rejects a reserve below 30 minutes. The original deadline is the termination boundary; the earlier cutoff is exported as `UE_COMPUTE_DEADLINE_EPOCH` and is the compute/projection boundary.

Lambda recommends using the PyTorch build shipped with Lambda Stack for
compatibility, and the public image documentation does not promise one immutable
PyTorch version. Accordingly, bootstrap accepts only PyTorch 2.5–2.x, while the
Python ML dependencies are exact released pins. The paid production preflight
exercises the exact Qwen3.5-9B training/reload path, and every checkpoint manifest
records the exact Python, PyTorch, CUDA, cuDNN, dependency, GPU, and optional
image-ID attestation. Keep Stage 1 and any authorized replication on the same
instance. If the project proceeds beyond feasibility, reproduce that attested
runtime in a pinned container before the paper-scale study.

## Arm the independent termination watchdog

On the **local Windows machine**, not the GPU, keep the machine awake and on reliable power/network. Store the Lambda API key only in the current process:

```powershell
$env:LAMBDA_API_KEY = Read-Host -MaskInput 'Lambda API key'
$hardDeadlineUtc = [datetime]'YYYY-MM-DDTHH:MM:SSZ'
./scripts/external_terminate_watchdog.ps1 -InstanceId INSTANCE_ID -HardDeadlineUtc $hardDeadlineUtc
./scripts/external_terminate_watchdog.ps1 -InstanceId INSTANCE_ID -HardDeadlineUtc $hardDeadlineUtc -Arm
```

Run the armed command in a separate terminal. This is a last-resort cap, not a substitute for monitoring, retrieval, console termination, and verification that instance status becomes `terminated`.

## Paid bridge preflight

```bash
bash scripts/preflight_bridge.sh configs/bridge_smoke.yaml configs/bridge_pilot.yaml
```

The smoke config uses the exact pinned Qwen3.5-9B text model and reward-learning
code in explicit non-thinking mode, with only 64 training worlds and one update.
It is not permitted to substitute a smaller model. It builds its own tiny data,
runs oracle mechanics, verifies the expected Qwen3.5 architecture and every
DeltaNet/attention/MLP LoRA target, trains one G-RL/P-RL pair, evaluates smoke
development data plus the unchanged base, writes timestamped logs and GPU
telemetry, and requires the smoke gate. It hash-verifies the transferred formal
corpus and the locally frozen token-workload profile used for projection, but it
does not parse or expose formal bridge DEV or locked TEST records. A passing run
writes `artifacts/bridge_qwen35_9b_pilot/PREFLIGHT_PASS.json`,
bound to the exact smoke report, smoke and Stage 1 configs, model revision,
architecture/adapter attestation, project source tree, formal frozen-data
manifest, and runtime identity.

The smoke acquisition diagnostic window is exactly its sole optimizer update.  In the formal config, terminal acquisition continuation gates are computed from the last 50 optimizer updates only (50 × 64 choices per arm), stratified by cue regime and aligned versus diagnostic-conflict trials.  Checkpoints also retain cumulative diagnostics for learning curves; cumulative performance is explicitly not the continuation-gate statistic.  Both states are checkpointed and verified on resume.

A pass validates installation, CUDA/BF16, non-thinking prompt rendering, text-only
model loading, model caching, complete hybrid-architecture LoRA coverage, paired
training, checkpoint reload/resume, evaluation, and analysis wiring. It is not
scientific evidence. Evaluation batch sizes, deterministic generation-subset
selection, one-token generation, legal-choice-mass and parse thresholds, and
bootstrap count are frozen in the config; runtime overrides that differ from
those values are rejected. Every prediction row carries the exact evaluation
spec/hash and the shared deterministic generation-subset attestation. Stop if any
loss is nonfinite, an adapter cannot reload, the pair is incomplete, or observed
peak memory or projected formal cost plus 30% does not fit the remaining budget.
The projection also reserves the exact 35 minutes of Stage 1 control-plane command
ceilings (10 minutes for the oracle, 20 for analysis, and five for the gate); it
is not silently treated as free time.

## Bridge Stage 1: one paired seed, DEV only

After preflight has populated the exact model cache:

```bash
bash scripts/run_bridge_stage1.sh configs/bridge_pilot.yaml configs/bridge_smoke.yaml
```

Stage 1 first re-verifies the hash-bound `PREFLIGHT_PASS.json` against the exact current project, configs, model revision, runtime, smoke report, and installed formal data. Any mismatch aborts before training. It then derives the first seed and both objectives from the frozen config, trains the paired G-RL/P-RL policies, evaluates only the configured development split, writes `artifacts/bridge_qwen35_9b_pilot/analysis/stage1_dev_report.json`, and requires the Stage 1 gate. Model loading is offline after preflight.

Before doing anything else, retrieve or inspect:

- reward-learning curves and whether both objectives were actually induced;
- the preregistered last-50-update acquisition gate cells, separately from cumulative learning-curve diagnostics;
- ordinary accuracy, action disagreement, and each seed's mean absolute `P(A)`
  gap between the pair;
- value- and transition-intervention fingerprints, no-switch controls, and shams;
- channel-role counterbalancing and update-comprehension checks;
- run manifests, peak VRAM, throughput, telemetry, wall time, and cost.

If Stage 1 fails, stop. Do not tune against or inspect locked TEST. The Stage 1 script ends after its DEV gate and cannot invoke replication.

## Human-gated replication and locked TEST

Continue only if Stage 1 is scientifically interpretable and projected remaining runtime plus 30% fits both the compute cutoff and dollar budget. Create the exact approval file manually:

```bash
printf '%s\n' 'I reviewed bridge DEV Stage 1 and authorize locked TEST replication' \
  > artifacts/bridge_qwen35_9b_pilot/APPROVE_BRIDGE_REPLICATION
bash scripts/run_bridge_replication.sh configs/bridge_pilot.yaml
```

Before any replication work, the script checks both the approval text and the Stage 1 report with `bridge-gate --require stage1`. It then trains the remaining paired seeds. Only after the complete paired matrix exists does it evaluate every seed on locked TEST, analyze the result, and require the replication gate.

The approval authorizes this single frozen replication. It does not authorize hyperparameter searches, controller-specific recovery, repeated looks at TEST, the symbolic SFT matrix, or a prospective reward-hacking phase.

## Symbolic apparatus calibration

The symbolic intended/proxy/cached organisms directly encode their controller in supervised targets. Their CPU oracle is useful for checking generation, intervention pairing, scoring, and analysis:

```bash
python -m under_extinction --config configs/smoke.yaml smoke
```

Do not run `run_stage1.sh` or `run_full_matrix.sh` on paid hardware as the default research plan. A later paper ablation may authorize a small symbolic model check, but a twelve-adapter SFT pass is not a substitute for the same-environment RL bridge.

## Retrieval before termination

```bash
bash scripts/collect_artifacts.sh
sha256sum deployment/results_*.tar.gz
tar -tzf deployment/results_*.tar.gz | head
```

Copy the result archive and checksum off the instance. Verify the SHA-256 at the destination and confirm it contains bridge configs, source, run/evaluation manifests, per-example predictions, reports, timestamped logs, telemetry, the final adapter's `bridge_state.pt`, and **every fixed bridge checkpoint** used for the learning-dynamics claim. Completed legacy calibration runs keep only their latest recovery checkpoint; bridge runs retain the full preregistered series, including optimizer/environment state. If any run is incomplete, confirm its latest resumable checkpoint is present; a partial adapter alone is not resumable.

Terminate through the console/API, then verify the instance reaches `terminated`. `shutdown` and `poweroff` do not terminate a Lambda instance and billing continues. Local instance data is destroyed on termination.
