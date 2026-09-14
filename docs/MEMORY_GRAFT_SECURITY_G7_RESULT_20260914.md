# G7 joint-pretraining result

Date: 2026-09-14

## Classification

G7 is **invalid for its routing estimand** because the registered trigger map
failed its installation gate in both matched arms. It therefore supports
neither planned central reading: it does not show that backbone routing
survives joint pretraining, and it does not establish joint pretraining as a
boundary condition that moves the write into memory. Target-row results are
likewise uninterpretable because there was no installed behavior to localize or
delete.

The pretraining portion supplies a narrower developmental positive. Conditional
memory was load-bearing from initialization in both executions: bypass-minus-
intact held-out NLL was 0.1340 with a 95% Student-t interval
[0.1182, 0.1498] in the source and 0.1334 [0.1193, 0.1475] in the replay.
At matched parameter count and tokens, conditional-minus-dense NLL was -0.01945
[-0.03600, -0.00289] and -0.01869 [-0.02753, -0.00984]. Because the frozen
exact-replay requirement failed, these observations remain developmental and
do not enter the manuscript's headline evidence.

## Frozen design and execution

The two approximately 166.927M-parameter Pythia-shaped models differ by 461
parameters. Each of three paired seeds per arm saw exactly 1,006,632,960 frozen
FineWeb-Edu tokens. The conditional-memory path was present from the first
optimization update; the dense control used an iso-parameter residual module.
The trigger, near trigger, and benign marker each occurred zero times in the
frozen pretraining split. Each source pretrain was repeated from scratch, then
both ordinary and frozen-component posttraining arms were run with symmetric
component transplants. Target, benign, and 16 random-row interventions applied
only to conditional memory; dense row deletion remained structurally
inapplicable because that arm has no rows.

The six source and six replay pretrains all completed. All 12 posttraining
invocations completed after a narrow pre-outcome implementation amendment. The
first source posttraining launch had attempted to tokenize numeric metadata in
the mixed marker object and stopped before optimizer construction,
optimization, evaluation, or intervention. The amendment selected only the
five frozen text fields; it changed no scientific choice. The failed log,
original receipt, amendment, amended runner, and new receipt are preserved.

## Installation failure

The registered per-seed installation threshold was 0.234881 attack excess. In
the source ordinary arm, conditional-memory values were [0, 0, 0.04004] and
dense-control values were [0, 0, 0.004883]. In the replay they were
[0, 0, 0.002930] and [0, 0, 0]. Every ordinary and frozen-component seed failed
the installation reading. The failure occurs in both architecture arms, so it
is an apparatus failure for the intended routing contrast rather than evidence
for either storage location.

## Replay failure

All 24 per-run manifests validate internally in the frozen verifier. However,
none of the six final pretraining checkpoints or twelve posttraining
checkpoints is tensor-exact across source and replay. The post-hoc divergence
audit finds a maximum absolute tensor difference of 0.595703 and as many as
98.56% of parameter values differing in a checkpoint. Across posttraining raw
prediction files, 135,974 of 221,184 rows differ. The source and replay still
produce the same categorical registered decision: memory is load-bearing,
installation is invalid, and routing is invalid for interpretation. This
qualitative agreement does not satisfy the prospectively frozen exact-replay
rule.

The short smoke replay was insufficient to establish billion-token exact
determinism. The runner seeded software RNGs but did not enable PyTorch's strict
deterministic-algorithm mode or its supporting CUDA workspace configuration.
This is a plausible source of divergence, not a post-hoc proof of the cause. No
third run or criterion change is used to rescue G7.

## Compute and artifacts

Source plus replay consumed 18.098 measured GPU-hours: 17.932 for pretraining
and 0.166 for posttraining. Including the approximately 22.70 hours used before
G7 and the 1.485-hour excluded benchmark gives approximately 42.283 GPU-hours,
below the 50-hour budget.

- Complete no-checkpoint archive SHA-256:
  `931a8ab116ff49633223d30f5c62711a167a9670904887c8e47cf5d9dd2c06e1`.
- Frozen verifier output SHA-256:
  `4d960ea375028a0c13cbf35a81f4e61b16160952164570fa7f052ba4cde38668`.
- Independent divergence audit SHA-256:
  `75da0af6f4b09f010ad70d44dcc0fe06e199eced9d57dfccf60827d1d2bd42dc`.

## Consequence for the paper

G7 leaves the paper's structural limitation open. It does rule out the narrow
concern that the jointly trained memory path was vestigial: the path had a
large, consistently positive held-out NLL contribution. What it could not do
was install the registered storage probe under the matched posttraining recipe.
The paper discloses this attempt in the limitations and evidence-history table,
keeps all G7 routing and row-locality claims out of the results and abstract,
and retains the existing thesis based on the fully verified retrofitted-memory
experiments.
