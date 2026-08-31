# Candidate: Visual Hindsight Leakage Across a Causal Cut

## PI status

**Backup candidate with a redesigned G0 v2; the CPU audit passed and the gate
awaits a clean commit freeze plus explicit GPU authorization.** G0 can establish
a necessary phenomenon, not an ICLR paper by
itself. A positive synthetic result must later survive natural-video edits,
multiple native-video checkpoints, and an automatic causal-cut method.

## Exact hypothesis

Each pair has one fixed visual past through a frame marked `CHOICE` and two
counterfactual futures. In one future the token moves to endpoint (e_0); in
the other it moves to the other admissible endpoint (e_1). The queried past
state and correct past answer are unchanged. A temporally valid model must
therefore answer the same past location in both worlds.

The primary estimand is not binary error. With three labeled locations, define

\[
  D_i = \tfrac12[1(a_{i0}=e_{i0})+1(a_{i1}=e_{i1})]
      - \tfrac12[1(a_{i0}=e_{i1})+1(a_{i1}=e_{i0})].
\]

This endpoint-assignment contrast is positive only when retrospective answers
track the experimentally assigned future more than the alternative future. A
constant answer bias cancels. G0 separately reports both counterfactual worlds,
all six ordered past-to-future transitions, and every frozen factor level.

## Why v1 was not launchable

The first scaffold used two boxes, so the only wrong answer was necessarily the
future endpoint. Its “directed” metric could not distinguish endpoint following
from generic degradation. Its stationary congruent future was not motion
matched to its moving incongruent future, and it could kill the hypothesis when
the model simply failed to perceive the suffix. It also passed ordered images
as a gallery rather than native video and did not bind the complete frame tree
into offline evidence. G0 v2 removes each of those failure modes.

## Frozen v2 corpus

G0 contains 48 exactly crossed synthetic pairs. Three equidistant labeled boxes
`A`, `B`, and `C` form a near-equilateral triangle. The token occupies a
factorially balanced past location at `CHOICE`. The two suffix worlds move it,
with the same frame count, interpolation schedule, and path length tolerance,
to the other two locations. Thus both futures contain target motion; neither is
a static low-salience control.

Each pair has five calls:

1. `prefix_past`: past-state question on the causal prefix;
2. `cf0_past` and `cf1_past`: the same past-state question after each future;
3. `cf0_future` and `cf1_future`: suffix-state positive-control questions on
   exactly the same full videos.

The last two calls establish that the model actually perceived each assigned
endpoint. A null retrospective effect is uninterpretable if these controls
fail.

The 48 cells exactly cross past location with token shape, background tone,
layout mirror, and token tone. Every pair contributes both counterfactual
endpoints, producing exactly 16 observations for each of the six ordered
past-to-future transitions. Prefix PNG bytes are identical across both worlds.
The evidence snapshot includes every frame, the input table, exact inventory,
per-frame SHA-256 values, prefix-identity commitment, and corpus-tree digest.

## Presentation and models

The primary condition uses Qwen3-VL-8B-Instruct's native `video` processor with
frame sampling disabled, so all frozen frames and temporal positions reach the
model. A Qwen multi-image condition is an optional presentation ablation.
Gemma 4 currently enters only as a later multi-image comparison because it is
not a like-for-like native-video replication; it cannot rescue or overturn the
primary decision.

The primary run contains 240 deterministic calls. The runner snapshots the
entire corpus before model loading, infers only from that snapshot, hashes it
again after inference, records code/config/git and runtime provenance, and
refuses to overwrite an existing root.

## Assay validity and gate

Before a scientific decision, all of these controls must pass:

- prefix past-state accuracy at least 75%;
- future-state accuracy at least 80% in each world;
- future-state accuracy at least 62.5% in every ordered transition; and
- strict-parser invalid rate at most 5%.

Failure returns `INVALID_ASSAY_DO_NOT_INTERPRET`; it never kills the hypothesis.

`EXPAND_VISUAL_HINDSIGHT_STUDY` requires all of the following:

- mean endpoint-assignment effect at least 25 points and bootstrap 95% lower
  bound above 10 points;
- endpoint-following rate at least 25% in each counterfactual world;
- endpoint-following at least 12.5% in every ordered transition;
- both worlds follow their assigned endpoints in at least 15% of pairs;
- assignment effect at least 10 points in every frozen factor level; and
- prefix-only inference repairs at least half of endpoint-following corruptions.

The hypothesis is killed only after the assay passes and the 95% upper bounds
exclude the frozen meaningful-effect regions for the assignment contrast and
both direct endpoint-follow rates. Results between pass and kill are
`INCONCLUSIVE_DO_NOT_RETUNE_THRESHOLDS`.

## Novelty boundary

Recent counterfactual and temporal-video benchmarks already own broad claims
that Video-LLMs struggle with event order, causal reasoning, and temporal
grounding. The surviving contribution is narrower: a within-prefix,
post-query intervention that measures whether an assigned future rewrites an
unchanged past answer. The closest references include EgoToM, MoMentS,
CounterVQA, VCRBench, Video-MME-Logical, REVEAL, CaST-Bench, and V-STaR. The
paper must not claim that counterfactual video evaluation, temporal cropping,
or three-way controlled animations are themselves new.

If G0 passes, the next experiment must use actor- and event-matched natural
video edits, belief/intent and physical-state questions, human temporal-validity
checks, native-video replication across model families, and learned or
automatically detected causal cuts. The simple crop is a diagnostic control,
not the method contribution.

## Integrity and launch boundary

The entrypoint is `scripts/run_visual_hindsight_g0_remote.sh`. It requires an
audited git commit and a stable SHA-256 over `src/visual_hindsight_g0/*.py`,
checks the frozen config digest, verifies the exact dependency/runtime preflight,
uses a non-overwriting run-root lease, and writes a terminal tree manifest only
after verification succeeds.

The config remains `cpu_validated__awaiting_commit_freeze_and_explicit_gpu_authorization`.
The focused CPU suite and shell syntax must pass again after the final commit
and source/config digests are frozen. Do not substitute a GPU result for those
prerequisites.
