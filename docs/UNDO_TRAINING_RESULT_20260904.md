# Corrected UNDO training DEV — verified result

Completed three arms in410.82seconds on GH200. One seed, synthetic register
tracking, native Qwen3-4B. No paper green light.

Evidence: retrieved/undo_training_20260904T0850Z, immutable inner run directory.
Archive SHA remote/local:
7fddf75458a36401f06c06ddf91c2b296f4ec65b8c194b4ec1d7f0e4cf3d6187.
Manifest0961b721a43b0bf9222b56d09883b935e688688f94ed9247706dfdeff4f3da9b;
58files verified, seeded update schedules and211916studenttokens/arm reproduced,
256updates per arm; initial/final/intermediate adapters and optimizers preserved.
This is evidence and accounting verification, not neural or optimizer replay.

## Result on depth60/100 histories

| Method | Correct /32 |
|---|---:|
| No adaptation |23|
| Terminal SFT |24|
| Canonical-state distillation |28|
| Local-rewrite distillation |24|

Local versus canonical loses four cases and wins zero: -12.5percentage points,
descriptive paired bootstrap[-25,-3.125]. Local versus SFT exchanges one win
and one loss, netzero. Local versus baseline wins one with no losses. These
small synthetic DEV intervals are not population or three-seed confirmation.

Initial DEV canonical, padding and counterfactual controls each8/8 and answer
mass~1. Final long-history padding itself has some errors, so do not attribute
every long-context error exclusively to obsolete-information interference.
All per-depth/control counts remain in verified.json, not just headline means.

Numerical checkpoint audit confirms actual parameter changes in every arm:
adapter L2changes .001122(SFT),3.075051(canonical),.004262(local). This is
consistent with a weak local training signal, but it does not identify a causal
mechanism or excuse the result. No learning-rate/checkpoint tuning after results.

## PI decision

Park this local-rewrite method; do not launch its three-seed expansion. The
stronger existing canonical-state comparator improves the tested task, so this
is not evidence all training fails. It does not establish a novel contribution:
canonical-context distillation is already prior work, and this experiment uses
one-action full-vocabulary KL rather than a full on-policy CCOPD reproduction.

The corrected algebra remains valid, but algebraic validity alone does not
ensure an informative training signal or length generalization. Preserve all
evidence and distinguish this negative method comparison from an invalid assay.
