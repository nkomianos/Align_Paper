# Stage C: existing repair baselines, conditional on actual bridge damage

Status: implemented with CPU component/integration tests, not run on GH200.
No natural sender update, paired update experiment or repair result exists yet.
These repairs are **baselines**, not our proposed scientific contribution.

## Entry condition

Run only after both fixed sender updates qualify and the locally verified paired
report says `REPEATED_EXTRA_LATENT_LOSS_REPAIR_STUDY_NEEDED`. The source protocol
now explicitly requires an actual >=10pp C2C accuracy decline as well as >=10pp
extra decline relative to text. This correction was made before any training or
paired outcomes: better text alone is not evidence that the bridge broke.

The runner takes the original qualification ticket and a separately checksum-
pinned paired report. Both must refer to the two predefined seeds. No private
answer keys are sent to the GPU. Fresh roots only; preserve all failed attempts.

## Calibration, separated from scored answers

Use the frozen 128 unlabeled repair examples (64 from each dataset). Their hash
is `9f2fa5d28ea80144de5e9260a5bfc446ea5a87cd7a90177c770d5f345d81f559`.
Within each dataset, rank IDs by SHA-256 of `repair-fit-v1:` plus the ID: first
48 are fitting data, remaining 16 are calibration checks. Thus 96 examples fit
and 32 check. No final-answer outcomes or labels select a method/hyperparameter.

Collect old-sender, updated-sender and receiver caches: 384 backbone forwards.
Use `prompt[:-1]`, exactly the source-prefill segment of the pinned C2C wrapper.
Sample at most 32 evenly spaced token positions per example, including endpoints.
Store B,H,N,D tensors for sender layers 8–35 and receiver layers 0–27, matched on
prompt/position. Calling the backbone without its vocabulary head produces the
same caches; a tiny native-model test verifies this.

These are **post-RoPE** keys and values. The affine maps below are not an exact
reproduction of RoPE-aware cache-transfer methods. That limitation matters for
position/length generalization. Preserve all selected caches for later audits.

## Fixed baselines

Fit key and value maps separately, independently in each head and layer:

| Arm | Fit or intervention |
| --- | --- |
| Identity | Unmodified input passed through the repair wrapper; implementation control |
| Diagonal | Affine per-feature least squares with ridge toward unit scale |
| Ridge (primary) | Affine head-local least squares, regularized toward identity |
| Orthogonal | Centered orthogonal Procrustes plus mean correction; reflections permitted |
| Fuser retuning | Fit a copy of each released projector to its original outputs on matched caches |

For diagonal/ridge, lambda is 0.001 times each head's mean centered feature
variance, with a 1e-8 variance floor. Solve in float64 and save float32 maps.
Apply maps to float32 cache values and cast back to BF16 before the unchanged
projector. Do not cast the map parameters to BF16 through wrapper construction.

Retuning uses 100 AdamW steps per layer, LR 1e-4, default betas/epsilon, zero
weight decay, gradient clipping 1.0 and 64 sampled cache-token positions per
step. Seed is the sender-update seed plus target-layer index. Fit float32 teacher
and student copies; keep eval mode and freeze hard gate logits. This disables
dropout/Gumbel noise and preserves the original gate pattern. Minimize equal
key/value output MSE, normalized by teacher-minus-receiver residual energy with
a 1e-6 floor. Save float32 and deployment-BF16 weights, all sample indices and
loss curves. This is output-matching adaptation, **not** a reproduction of full
task-loss fuser training or a newly proposed method.

Report held-out calibration cache/output errors without selecting a winner or
changing hyperparameters. There is no parameter sweep or checkpoint selection.

## Language evaluation and analysis

For each seed, evaluate all five arms on the same reserved 256 questions:
**1,280 additional generations per seed**, 64-token greedy cap. Rotate arm order
by case index. Preserve messages, token IDs, timing and truncation. The unchanged
receiver, sender-alone, text-transfer and unmodified bridge results come from
the fully reverified paired run.

Require identity-wrapper token agreement >=98% with the corresponding unmodified
updated bridge and >=95% parsing in every repair arm. Primary ridge recovery is

`(ridge accuracy - damaged bridge accuracy) / (original bridge accuracy - damaged bridge accuracy)`.

The denominator must be positive. Require >=75% recovery and a paired-question
bootstrap lower endpoint above zero for ridge's improvement. Use 5,000 resamples,
stratified by dataset, seed 202609044. Both sender-update seeds must meet this;
do not promote a successful secondary method to replace a failed primary.
Always report secondary methods, including a stronger retuned-fuser result.

Cost accounting includes original-sender retention, all three calibration
backbones, map fitting/checking/serialization, fuser fitting, and generation.
Report per-method preparation times and a conservative amortization bound that
charges all baseline preparation. Separate-run serial timings are not an optimized
serving benchmark; no speed claim follows solely from token counts or fitting loss.

## Verification and launch

Entry points are `scripts/run_c2c_cache_repair.py` and
`scripts/verify_c2c_cache_repair.py` (`--help` lists exact arguments). Use the same
isolated torch 2.7.1+cu128 / Transformers 4.52.4 environment and fresh committed
checkout. The verifier re-verifies both paired parents, source, complete recursive
digests, data partitions, cache shapes, sampled positions, and repair output grids.

Closed-form maps are independently refitted on CPU. Ridge/diagonal coefficients
must match numerically. Orthogonal solutions can be nonunique, so verification
checks orthogonality, centering and optimal calibration error rather than forcing
the same arbitrary SVD basis. Retuning telemetry/checkpoints are checked but its
optimizer is **not** independently replayed; do not claim otherwise.

Even successful recovery yields `EXISTING_REPAIR_RECOVERY_REPEATED_NOVELTY_UNRESOLVED`,
not a paper green light. The study would still need a non-obvious and important
deployment finding, independent models/tasks, comparisons to the nearest methods
under their actual protocols, and a reason the contribution exceeds known adapter
incompatibility. No automatic paper expansion follows this pilot.
