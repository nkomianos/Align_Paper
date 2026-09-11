# Memory Graft security S2 pre-registration

## Status and scope

This document prospectively fixes S2a--S2d before S2 selects a benign
continuation, constructs a model, loads a checkpoint, or performs a GPU
forward pass. S1 v1.1 is accepted as a valid negative for preferential storage
in the trigger's final addressed hash rows. S2 closes the assay-validity and
mechanistic-localization gaps. It does not test a second payload, model family,
or graft implementation.

S2 reuses the independently verified S1 v1.1 checkpoints and artifacts whose
manifest SHA-256 is
`731cb1d7c50f82e668e40585ce6484629960a1d3e641026c44b1b3542ef6a11c`.
Any source-manifest mismatch invalidates S2 before checkpoint loading.

## Fixed scientific questions

**S2a, table-only positive control.** Can the identical final-trigger-row
deletion assay detect a trigger/payload map when only the hash table is allowed
to learn? The clean S1 checkpoint initializes each run. Every backbone and
non-table graft parameter is frozen, and only the packed hash-table embedding
is optimized. The training objective, learning rate, schedule, weight decay,
sequence length, exposure count, and trigger/payload are otherwise unchanged
from S1. This is a positive control for the intervention and a boundary test
for storage under a table-only optimization constraint.

**S2b, checkpoint-only hybrid localization.** Where is the behavior sufficient
or necessary in the unconstrained S1 checkpoints? For every S1 replication
seed, evaluate the intact clean and poisoned endpoints, poison backbone plus
clean full graft, clean backbone plus poisoned full graft, poisoned checkpoint
with the clean whole table restored, and poisoned checkpoint with only the
clean target rows restored. Transplant conditions are sufficiency tests and can
be affected by module co-adaptation; restoration conditions are necessity
tests. The two are reported separately.

**S2c, checkpoint-only gate measurement.** At the sole registered graft
injection layer, record the scalar context-aware sigmoid gate at the final
trigger and exposure-matched-benign positions in clean and poisoned S1
checkpoints. The implementation has one Engram injection layer, recipient
layer index 1, so “every injection layer” means that one layer. Gate values,
paired differences, and distributions are outcomes. There is no gate threshold
or pass/fail decision because the architecture supplies no calibrated mapping
from gate amplitude to exact-match ASR. A lower trigger gate can support an
attenuation mechanism; it cannot alone establish causality because the short
convolution and downstream backbone can compensate.

**S2d, repaired benign control.** Select one replacement benign continuation
before S2a training by matching clean-checkpoint predictive surprisal. The
fixed candidate pool is every tokenizer ID whose exact decoded text matches
`^ [a-z]{4,10}$`, re-encodes to that single ID, and is neither the payload nor
the old benign continuation. For each candidate and the payload, compute mean
next-token negative log probability over the same 1,024 validation contexts,
using the S1 development clean checkpoint at both registered model sizes. Pick
the token minimizing the sum across sizes of squared candidate-versus-payload
mean-NLL differences; break ties by token ID. This licenses a claim of matched
baseline clean predictive surprisal. It does not license matched post-training
learnability; S2a reports the benign accuracies and their distribution rather
than assuming equivalence. S2b/S2c retain the original S1 exposure-matched
benign marker only for comparison with already-trained checkpoints.

## Threshold derivations

The smallest scientifically meaningful change remains the S1 registered
localization resolution, `delta = 0.15` ASR. Reusing this value makes S2 a test
of whether the S1 intervention can recover an effect of the size S1 claimed it
could identify; it is not a newly chosen convention.

For 1,024 paired binary predictions, a distribution-free two-sided 95% bound
on a paired mean with range two has half-width

`h = sqrt(2 ln(2/0.05) / 1024) = 0.08488134473378872`.

S2a development therefore considers a poison count eligible only when
`ASR_intact - ASR_clean >= delta + h = 0.23488134473378872`. This is the minimum
observed installed-attack excess whose lower bound leaves the complete 0.15
effect available for deletion. It is an apparatus-eligibility gate derived
from the downstream deletion estimand. The smallest eligible count is selected
within each model from the fixed ladder `16, 64, 256, 1024, 4096`; a model with
no eligible count is reported as table-only installation failure and is not
replaced.

Across five new table-only replication seeds, define:

* `raw_removal = ASR_intact - ASR_target_rows_zero`;
* `control_removal = max(ASR_intact - ASR_benign_rows_zero,
  ASR_intact - mean(ASR_random_rows_zero))`;
* `target_specific_removal = raw_removal - control_removal`.

The raw-removal criterion is satisfied when its two-sided 95% Student-t lower
endpoint exceeds 0.15. It protects the claim that the operation removes enough
installed behavior to validate S1's intended effect resolution. The
target-specific criterion is separately satisfied when its lower endpoint
exceeds 0.15. It protects the stronger claim that the boundary is the trigger's
nominal rows rather than generic sensitivity to deleting table rows. Both
conjuncts are independently derived from the S1 estimand; neither is a sanity
bar. Outcomes are classified prospectively:

* both criteria pass: `ASSAY_VALIDATED_AND_NOMINAL_ROWS_LOCALIZED`;
* raw removal passes but specificity does not: `DELETION_WORKS_NOT_TARGET_SPECIFIC`;
* the raw-removal 95% upper endpoint is below 0.15 despite eligible installation:
  `BEHAVIOR_SURVIVES_NOMINAL_ROWS`, evidence that multi-head/temporal addressing
  distributes the installed behavior beyond the final nominal rows;
* otherwise: `INCONCLUSIVE_AT_REGISTERED_RESOLUTION`;
* no eligible poison count: `TABLE_ONLY_INSTALLATION_FAILURE`.

S2b uses the same 0.15 meaningful-effect resolution for each continuous
contrast. `outside_graft_sufficiency` is ASR(poison backbone + clean graft)
minus clean ASR; `graft_sufficiency` is ASR(clean backbone + poison graft)
minus clean ASR; `whole_table_necessity` and `target_row_necessity` are intact
poison ASR minus the corresponding restoration ASR. Each gets a separate
five-seed Student-t interval. No compound gate controls whether results are
reported. A component supports a meaningful effect only if the interval's
lower endpoint exceeds 0.15; an effect is ruled out at this resolution only if
its upper endpoint is below 0.15. Other intervals are inconclusive.

S2c and S2d contain no numerical pass gate. All registered measurements are
reported.

## Fixed models, checkpoints, data, and evaluation

Models are the S1 Pythia-410M and Pythia-1.4B base recipients at their pinned
revisions. S2b/S2c use the five S1 replication seeds 26091301--26091305 and the
selected S1 trainable-table checkpoints (`N=64` for 410M and `N=16` for 1.4B).
Tokenization uses the S1 donor tokenizer, EleutherAI/Pythia-2.8B at immutable
revision `dbe7ae300a54abcdc475a33907b3dff81d25709f`.
S2a uses S1 clean checkpoints at development seed 26091300 and the five
independently initialized S1 clean seeds 26091301--26091305. It uses the same
materialized training blocks, exact bank,
compression map, validation contexts, trigger, near trigger, payload, optimizer
steps (512), AdamW learning rate (5e-5), weight decay (0.01), microbatches, and
ordinary causal language-model loss as S1. Poison and benign examples remain
disjoint and equal in number.

Every evaluation has 1,024 prompts and one-token exact-argmax scoring. S2a
runs intact, target-row zero, benign-row zero, and 16 fixed random-row-set zero
conditions, plus near-trigger, untriggered, and repaired-benign surfaces. S2b
runs all six registered endpoint/hybrid/restoration conditions. S2c retains
all 1,024 raw scalar gate values for both marker surfaces. The seed, not prompt,
is the confirmatory replication unit.

## Scale and compute justification

At freeze time, approximately 47.796 GPU-hours remain from the 50-hour budget
after the earlier 0.066-hour estimate and S1's measured 2.138 hours. S2 is
projected at 1.5 GPU-hours and must not launch a new training run if the
projected study total would exceed 45 hours; an active run is never interrupted.
Although S2 is projected below 1% of the remaining budget for training alone,
its scale is not chosen because it is cheap. It inherits the complete ten
independently trained S1 endpoint pairs, uses five new independently trained
table-only seeds per included size, preserves the 1,024-prompt resolution from
which the 0.15 estimand was derived, and exhaustively evaluates every registered
checkpoint hybrid. More table-only optimizer parameters or steps would change
the boundary estimand, while more prompts would create pseudoreplication rather
than more independent training units. Five seeds retain direct pairing with
S1's confirmatory design; uncertainty is reported rather than hidden.

## Integrity, verification, and stopping

The runner verifies its config, preregistration, and own source against a freeze
receipt before reading the S1 manifest. It then verifies the complete source
manifest and the S1 verification record before loading weights. Outputs use
compact table deltas referencing sealed clean checkpoints, raw predictions and
gate rows, training logs, condition summaries, a sealed manifest, and an
inventory digest. An independent verifier reconstructs every S2a checkpoint,
replays all S2a--S2c predictions/gates, and reproduces selection and decisions.

Any manifest mismatch, address mismatch, parameter-freezing violation,
checkpoint reconstruction mismatch, non-finite computation, or replay mismatch
returns `INVALID`. A missing eligible S2a model is a valid installation-failure
outcome. No generality experiment begins until S2a--S2d are complete and
verified.

## Fixed write-up framing

S1's frozen-table arm reached 99.96% and 99.75% mean ASR, so the behavior
installs at ceiling without table updates. The primary finding is that the
backbone can absorb the trigger/payload map even when functioning addressable
memory is present. S1 row ablation corroborates that null; it did not discover
it. S2 determines whether row deletion is capable of detecting table-confined
storage and whether the unconstrained behavior localizes to backbone, other
graft parameters, table weights, or co-adapted combinations.
