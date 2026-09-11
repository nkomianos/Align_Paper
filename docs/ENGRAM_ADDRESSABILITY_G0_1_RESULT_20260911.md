# Engram addressability G0.1 result

## Decision

**`S0_HARNESS_FAILURE_STOP`.** Calibration passed at 100% default accuracy.
After cloning that checkpoint and freezing every non-memory parameter, the
bigram address arm reached 82.30% held-out-context entity accuracy and the
current-token address control reached 38.87%. The 43.43 percentage point
registered advantage exceeded the 20 point contrast criterion, but the bigram
arm missed the separately required 90% absolute accuracy criterion. The frozen
S0 gate therefore failed and S1 was never run.

The large within-run contrast is a developmental observation that the registered
address change mattered after backbone freezing. It is not a passing result and
does not establish a poisoning vulnerability, causal localization, or ablation
remedy. Because the amendment allowed only one repair, this synthetic apparatus
is now closed. The proposed security mechanism remains untested at the decisive
S1 estimand.

## Execution and result

- Base frozen design commit: `3d118a289a6c658fc35c37d5b41a74b21eac347a`.
- Frozen amendment commit: `e76f54066440132bda325586a5376e08db5e5e58`.
- Amendment receipt commit: `b153a1e7f25bc5431cf60af988c5df802ccc5b4c`.
- Execution commit: `fc3ff551857bd4188a246beccfd833f92236bfea`.
- Runner SHA256:
  `29c2eac6777f3e9adff8745f7ee81444bbd4b3b83328f7ea766c55d896be63e3`.
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Calibration: 1,107,140 total parameters, 828,164 trainable non-memory
  parameters, 256 optimizer steps, 100% default accuracy.
- Each S0 arm: 1,107,140 total parameters, 262,336 table parameters, 278,976
  trainable memory-module parameters, and 512 optimizer steps.
- Bigram address arm: entity 82.2998%, default 100%.
- Current-token address control: entity 38.8672%, default 100%.
- Registered entity-accuracy advantage: 43.4326 percentage points.
- Total runner wall time: 7.0870 seconds.
- Never run: both poisoned S1 arms, exact trigger-row ablation, random-row and
  benign-row ablations, collision attacks, additional seeds, poison-count
  ladders, pretrained-backbone integration, and model-family transfer.

## Verification

Both the remote verifier and an independent local replay passed. The verifier
regenerated the registered datasets, checked the frozen amendment and its
provenance, reloaded the calibration and S0 checkpoints, replayed 16,384 raw
evaluation rows, checked 256 calibration and 1,024 S0 training-log entries,
confirmed equal S0 parameter counts, and matched all 22 manifested files.

The local evidence root `artifacts/engram_addressability_g0_1` contains 24 files,
including the manifest and completion marker. Its sorted inventory file has
SHA256
`eebe65f41724bd66421b1c16119ba6b9825a0d2f64023686e1abaaed205cd479`.
The remote verifier report
`artifacts/engram_addressability_g0_1_verified.json` has SHA256
`1cb1b75210aee2dc25e332b85d7e2147aa0b5d338b3287dd839cddc1ea61b81d`.
The local replay produced the same report content and hash.

## Interpretation and closure

G0 was invalid because a fully trainable Transformer memorized both address
arms. G0.1 removed that route and produced the registered directional contrast,
but it did not meet the absolute learnability requirement. Calling this a
positive would discard the frozen gate after seeing the result. Calling it a
negative about Engram security would also be wrong because the causal S1 assay
never executed. It is a valid developmental apparatus failure with a promising
unconfirmed signal.

No further tuning of this 128-pair synthetic task is authorized. Any future test
of deterministic-memory security must be a separately motivated and
preregistered experiment on a faithful pretrained implementation; it cannot be
presented as continuation or rescue of G0.1.
