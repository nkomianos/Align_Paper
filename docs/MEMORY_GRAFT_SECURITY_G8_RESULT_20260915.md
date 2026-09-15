# G8 inherited-checkpoint routing successor: final result

Date: 2026-09-15

## Classification

G8 produced a **valid negative unconditional-calibration result** and therefore
closed before its conditional-routing stage. Routing was **never run**. G8
supports no claim about whether joint pretraining routes later writes into the
backbone or conditional memory.

This is the registered outcome when the inherited checkpoints cannot learn the
unchanged payload unconditionally. It leaves the paper's jointly pretrained
structural limitation open and leaves the verified retrofitted-memory thesis
unchanged.

## Frozen marker and calibration

The new trigger was ` silent river signal`. Its token IDs 11561, 8281, and 2625
occur 11,379, 67,393, and 51,782 times in the frozen 1.0066B-token FineWeb-Edu
stream, while the complete phrase occurs zero times. Trigger selection and all
thresholds were frozen before any inherited checkpoint was loaded.

Unconditional exact-match accuracy gains across the three conditional-memory
then three dense-control checkpoints were:

| cell | exposures | steps | six source gains | all-six pass |
|---|---:|---:|---|---|
| C1 | 64 | 512 | 0, 0, 0, 0, 0, 0 | no |
| C2 | 512 | 512 | 0, 0, 0, 0, 0, 0 | no |
| C3 | 2,048 | 512 | 0, 0, 0, 0.0009766, 0, 0 | no |
| C4 | 4,096 | 1,024 | 0, 0, 0, 0.0009766, 0, 0 | no |

The derived gate was 0.15 for every checkpoint: the routing assay seeks a
15-point component contrast, so the apparatus must first demonstrate at least
a 15-point unconditional learnability gain. The maximum observed gain was less
than one tenth of one percentage point. The frequent-token trigger hypothesis
therefore did not rescue the apparatus; unconditional adaptation itself was the
blocking capability under this recipe. This result does not distinguish a
payload, learning-rate, token-budget, or general undertraining explanation.

## Verification and amendments

All 48 completed short calibration invocations reproduce every emitted final
state-tensor digest and prediction bitwise. Every source and replay manifest
validates. The inherited G7 checkpoints retain their disclosed non-bitwise
pretraining provenance; G8 treats the six source checkpoints as digest-pinned
inputs and does not weaken G7's replay criterion.

Two narrow pre-outcome harness amendments are preserved:

1. The first C1 run stopped after evaluation but before emitting a result because
   the state-hash helper did not serialize scalar buffers. Reshaping tensors to
   one dimension fixed only serialization.
2. The first C4 run stopped before optimization or evaluation because a 12M
   offset left too few tokens. C4 alone, and the conditional stage had it
   advanced, use the already sealed 10M offset. C1--C3 were not rerun.

Neither amendment changed a scientific outcome, model, marker, optimizer,
threshold, ladder cell, or selection rule.

## Compute and artifacts

The 48 completed calibration invocations used 0.3740 end-to-end GPU-hours. The
two stopped launches add approximately 0.0080 hours, giving about 0.382 G8
GPU-hours and approximately 42.665 cumulative GPU-hours for the program.

- calibration decision SHA-256:
  `ce7d506c118fc7927b8e0a209befb7eed6ad10f5e557e789eb37210d843cd62a`
- exact verifier SHA-256:
  `9bd2f2b4aa13c690b22e72e911a88c277b6eb2e06720910c70c8958a06735340`
- complete archive SHA-256:
  `7d24d7a9446b0963919f05c5444ed8e12255785b8c2c1bdd956ae7399399e033`

## Consequence for the paper

The abstract, primary results, and conclusion remain based on the verified
retrofitted-memory experiments. Section 6 reports the developmental G7 bypass
cost and states that it makes the joint-pretraining limitation more live. The
appendix evidence history classifies G8 as a valid negative calibration with
routing unrun. No G8 routing or locality claim is made.
