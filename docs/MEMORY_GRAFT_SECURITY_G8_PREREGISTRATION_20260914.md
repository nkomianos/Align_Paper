# G8 preregistration: inherited joint-pretraining routing successor

Status: frozen before any inherited G7 model weights were loaded by G8.

## Question and inherited scope

G8 is a separately motivated successor to the invalid G7 routing assay. It asks
only where a newly installed conditional mapping is stored after joint
pretraining. It reuses the six digest-pinned G7 **source** checkpoints: three
conditional-memory and three iso-parameter dense-control models, each pretrained
on 1,006,632,960 tokens. Pretraining is inherited and will not be repeated.

The G7 source and replay did not reproduce bitwise. G8 therefore treats the
source checkpoints as fixed experimental inputs and inherits only their
categorical provenance: the conditional path was load-bearing and the G7 probe
failed installation. G8 makes no new pretraining or converged-model claim.

## New marker fixed from the frozen token stream

The leading explanation for G7's installation failure is that the constituent
tokens of `Kavanaugh Galois Zygmund` were poorly trained in a one-billion-token
run. Before loading any inherited weights, we selected ` silent river signal`
using only the frozen FineWeb-Edu token array and pinned tokenizer. Its token IDs
and exact frequencies among 1,006,632,960 pretraining tokens are:

| surface | token ID | count |
|---|---:|---:|
| ` silent` | 11561 | 11,379 |
| ` river` | 8281 | 67,393 |
| ` signal` | 2625 | 51,782 |

Every constituent appears at least 10,000 times, while the complete three-token
phrase occurs zero times. The one-character near trigger ` silent river
signals` (final token 6298, count 36,208) and benign marker ` silent river
window` (final token 3497, count 39,127) also occur zero times. Trigger, near,
and benign final-position row sets have no overlap. The original one-token
payload ` quartz` and benign continuation ` purple` are retained so the
successor changes the suspected trigger representation rather than both sides
of the mapping.

## Stage A: unconditional calibration

No conditional cell may run until calibration passes. Each of the six inherited
checkpoints is restored independently and trained to predict the payload
unconditionally at a fixed token position in a subset of WikiText blocks. The
ordered ladder is:

| cell | payload exposures | optimizer steps |
|---|---:|---:|
| C1 | 64 | 512 |
| C2 | 512 | 512 |
| C3 | 2,048 | 512 |
| C4 | 4,096 | 1,024 |

All cells use length 256, microbatch 16, fused AdamW at 5e-5, weight decay
0.01, gradient clipping at 1.0, and 1,024 fixed held-out contexts. The first
cell at which all six checkpoints gain at least 0.15 exact-match accuracy over
their own pre-adaptation baseline is selected. Every new state hash and
prediction must also reproduce bitwise. Larger ladder cells are not launched
after the first passing cell. If no cell passes, G8 closes without conditional
training and the paper ships on the retrofitted evidence.

The 0.15 calibration gate is derived from the downstream routing estimand. A
component contrast of 0.15 cannot demonstrate storage of a learned mapping if
the checkpoint cannot gain at least 0.15 accuracy on the payload at all. No
additional prompt-level sampling allowance is added: the 1,024 contexts are the
complete frozen evaluation set and are not treated as independent training
replicates.

## Stage B: conditional routing

If calibration advances, every checkpoint receives the first passing cell's
exposure count and optimizer-step budget. Trigger/payload and benign/control
exposures are equal. Both ordinary and inserted-component-frozen adaptations
run. Symmetric transplants measure the clean component with poisoned outside
state and the poisoned component with clean outside state. Conditional memory
also receives whole-table and target-row restoration/sufficiency, target-row
zeroing, benign-row zeroing, and sixteen random-row controls. Dense row deletion
is structurally inapplicable and cannot be called a failure.

The central routing estimand is ordinary conditional-memory outside-component
sufficiency minus inserted-component sufficiency. The minimum meaningful effect
is 0.15, inherited from the deletion claim: the proposed storage component must
carry at least fifteen percentage points more exact-match behavior than its
control. With valid apparatus, a seed-level 95% interval entirely above +0.15
means backbone routing survives joint pretraining; an interval entirely below
-0.15 means memory routing and locates retrofitting as a boundary condition.
Anything between is mixed or indeterminate.

The installation gate is separately derived from the same estimand. Both
architectures' ordinary installed-attack-excess 95% lower bounds must exceed
0.15. Without that much installed behavior, a routing contrast of the registered
magnitude is not interpretable. This is the sole apparatus conjunct after the
calibration pass; memory load-bearing was measured in G7 and is inherited as
developmental provenance rather than retested.

The nominal-row security call retains three independently necessary readings:
target-row necessity protects removal, target-row sufficiency protects
containment, and target-row-zero specificity protects against benign or random
row disruption. Each 95% lower bound must exceed 0.15. These are reported
separately from the routing decision.

## Statistics, replay, and compute

The independent unit is the inherited pretraining seed followed by its complete
short adaptation, with three seeds per architecture. Two-sided Student-t 95%
intervals use `t(0.975,2)=4.3026527297`. Prompts estimate fixed-checkpoint
outcomes and are not inferential replicates. Calibration is a fixed ordered
selection rule; the routing contrast is the sole central estimand, and all
controls are reported.

G8 enables `torch.use_deterministic_algorithms(True)`, sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and disables TF32 before any model
construction. All new calibration and conditional runs require bitwise-identical
state-tensor hashes and raw predictions under full short-run replay. The
inherited G7 checkpoints themselves retain their disclosed non-bitwise
pretraining provenance; G8 does not weaken or relabel G7's failed replay.

Measured cumulative use before G8 is approximately 42.283 of 50 GPU-hours. The
worst case is 48 short calibration runs plus 12 conditional runs and their
interventions, conservatively bounded at 1.5 GPU-hours. The scale uses all six
available independent checkpoints and exact replay. Reducing it would remove
the seed-level routing interval, while repeating pretraining would add no
information to this inherited-checkpoint estimand.

Both valid routing directions are reportable. Persistent backbone routing
strengthens the central paper claim. Memory routing identifies retrofitting as
a boundary condition. A mixed result is reported without forcing either
conclusion.
