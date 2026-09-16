# G12 preregistration: deletion durability and collision isolation

**Pre-execution state.** No G12 weight has been loaded and no G12 neural
forward or optimizer step has run. The fixed collision was constructed solely
from the tokenizer, frozen compression map, and public hash function. This
protocol and its runner are hash-bound before execution.

## Question

S2e showed that a direct row-write API stores the mapping in 16 known rows and
that zeroing those rows removes 98.3--100% of the behavior. G12 asks whether
that deletion survives the next legitimate write. It tests two separate
failure mechanisms:

1. **optimizer-state remanence:** zeroing weights may leave Adam's first and
   second moments, which can move a deleted row again even when the next item
   addresses disjoint rows;
2. **hash-collision isolation:** another raw prompt may address the same rows,
   allowing its subsequent write to recreate or overwrite the deleted item.

These are independently interpreted. A failure of one does not gate the other.

## Frozen collision and controls

The original prompt ends in raw token `mund` (id 27778). The collision prompt
inserts a token boundary and ends in ` mund` (id 35271). Vocabulary compression
maps both to id 18985. Their complete compressed suffixes are identical, so all
eight order-2 and all eight order-3 table heads address the same packed rows at
both model sizes. The prompts remain different raw token sequences. The
existing `Kavanaugh Galois Quasar` marker overlaps zero target rows and is the
disjoint control. Exact row IDs are frozen in the JSON config and rechecked by
the runner.

## Procedure

For each of five existing clean checkpoints at 410M and 1.4B, train only the
original prompt's 16 rows for 512 Adam steps at the S2e-selected rate `1e-3`.
Retain the final row values and Adam state on those rows. Verify that all other
table rows remain bitwise identical.

Then restore the same installed state for six branches. Zero the 16 row weights
in every branch. In half, leave Adam's moments intact; in half, also zero both
moments on those rows. Apply one of three 512-step direct writes:

- disjoint marker to the original payload;
- fully colliding marker to the original payload;
- fully colliding marker to a different payload.

The same-payload disjoint arm controls for learning the same output elsewhere.
The same-payload collision arm is the direct cross-user resurrection test. The
different-payload collision arm measures overwrite and cross-talk rather than
presuming that every colliding user wants the same continuation.

Measure the deleted mapping after 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512
steps. All fixed times report exact match, target rank/MRR, log-probability, and
target-row L2. The endpoints use all 1,024 contexts and retain raw rows;
intermediate trajectories use the first fixed 256 contexts. The secondary
mapping and final clean NLL are also reported.

## Derived decisions

The downstream security effect is a 0.15 ASR change. At 1,024 paired contexts,
the registered Hoeffding allowance is 0.084881, giving a detectable per-seed
point effect of 0.234881. Initial installations below that value cannot expose
the downstream contrast and remain explicit apparatus failures.

The optimizer-remanence estimand is peak resurrection under a disjoint update
after weight-only deletion minus the same quantity after weight-plus-state
deletion. A five-seed lower 95% Student-t endpoint above +0.15 establishes that
optimizer state is part of the deletion boundary.

The collision estimand is peak resurrection after full-collision same-payload
updates minus disjoint same-payload updates, both after weight-plus-state
deletion. A lower endpoint above +0.15 establishes a collision isolation
failure independent of stale optimizer moments. The different-payload arm is a
separate, fully reported interaction outcome.

A branch supports durability only when the upper five-seed endpoint for peak
resurrection is below 0.15. Failure to reject resurrection does not establish
durability. Exact-match decisions cannot be replaced by rank or log-probability,
which diagnose metric sensitivity.

## Scale and verification

Five seeds match the validated S2e inferential unit. The six branches are a
fixed three-by-two mechanism design, and the trajectory is needed because Adam
remanence can be transient. Projected source plus exact replay is eight GPU
hours. Full replay, manifests, checkpoint hashes, optimizer logs, endpoint raw
predictions, and decision reproduction are required.
