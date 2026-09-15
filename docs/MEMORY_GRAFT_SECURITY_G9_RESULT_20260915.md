# G9 bounded calibration repair: final result

## Decision

G9 is a **valid negative capability calibration**. It does not identify a
joint-pretraining routing direction. The conditional routing, transplant, and
row-deletion stages were never run. Under the frozen stop rule, repair of the
six G7 jointly pretrained checkpoints is permanently closed.

All 36 registered calibration invocations completed across three learning
rates, two matched architectures, three independent inherited pretraining
seeds, and source/replay. All 18 source/replay pairs reproduce their state
hashes, continuous measures, training logs, and predictions bitwise.

## Repairs applied together

- Payload: one-token ` river` (ID 8281), observed 67,393 times in the frozen
  1.0066B-token FineWeb-Edu stream.
- Position: the payload is training token 64, predicted from logit position 63,
  exactly matching evaluation after a 64-token context.
- Exposure schedule: one payload row in every microbatch, 512 total exposures
  over 512 optimizer steps.
- Rates: 5e-5, 5e-4, and 1e-3, fixed and fully executed.
- Instrumentation: every step records payload-specific loss, rank/MRR,
  log-probability, and full-LM loss. Full 1,024-context rows and every
  post-calibration checkpoint are retained.

## Outcomes

| Learning rate | Exact-gain range | MRR-gain range | Mean log-probability-gain range | Post median-rank range | Gate |
|---:|---:|---:|---:|---:|---|
| 5e-5 | 0--0.0009766 | 0.000391--0.001501 | -0.206--0.215 nats | 3,112.5--4,628 | fail |
| 5e-4 | 0 in all six | 0.003667--0.006166 | 2.893--3.365 nats | 236--350 | fail |
| 1e-3 | 0 in all six | 0.006642--0.010443 | 4.130--4.515 nats | 117--145 | fail |

The exact advancement gate was 0.15 gain in every checkpoint. The rank-based
secondary criterion was 0.15 MRR gain in every checkpoint. No rate approached
either. The higher rates clearly moved the payload upward, so saying that the
model did not adapt would be false. The movement remained far too weak to make
the exact-match routing contrast measurable. Under the frozen classification,
this is capability failure rather than argmax-only metric sensitivity.

At 1e-3, mean full-LM loss increased from 5.122 on the first step to 5.564 on
the last, while mean payload-specific loss fell from 12.876 to 7.929. This
supports the same narrow interpretation: the payload received a learning
signal, but the aggressive global rate also degraded the general objective and
did not install the behavior.

## Verification, runtime, and evidence boundary

`VERIFICATION.json` passes all 18 source/replay comparisons. The 36 complete
invocation intervals sum to 1,004.78 seconds, or 0.2791 GPU-hours. Adding the
previous measured 42.665 gives approximately 42.944/50 GPU-hours.

The result rules out the complete G9 repair bundle as a usable routing assay on
these checkpoints. It does not show whether a better-trained jointly pretrained
memory model would route to memory, and it supplies no evidence for backbone or
memory routing. The paper retains the structural limitation.
