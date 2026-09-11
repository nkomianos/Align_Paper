# Engram addressability G0 result

## Decision

**`S0_HARNESS_FAILURE_STOP`.** The parameter- and operation-matched bigram and
current-token address arms both reached 100% accuracy on the held-out-context
entity-pair task and the ordinary default task. The registered 20 percentage
point address-mechanism contrast was therefore zero. S1 was never run.

This result says the S0 task failed to distinguish memory addressing from
backbone memorization. It provides no evidence for or against closed-form
backdoor localization.

## Execution and result

- Frozen design commit: `3d118a289a6c658fc35c37d5b41a74b21eac347a`.
- Execution commit: `7e7a7bfa87e3e0ee5a8ff65e399a102c99a38900`.
- Runner SHA256:
  `56a0931cfdfcdd6e0bd035b31aa77e2e22d11b76a54fc9b6af72e942cc07e44b`.
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Each arm: 1,107,140 parameters, including 262,336 table parameters; 512
  optimizer steps.
- Bigram arm: entity 100%, default 100%.
- Current-token control: entity 100%, default 100%.
- Total runner wall time: 7.8787 seconds.
- Never run: poisoned S1 arms, row ablations, collision attacks, poison-count
  ladder, additional seeds, pretrained-backbone integration, and model-family
  transfer.

The final 16-step mean training losses show a transient speed difference that
was not a registered outcome: at steps 64 and 128 they were 0.4863 and 0.0534
for bigram addressing versus 0.6005 and 0.2284 for the control. Both arms were
effectively at zero by step 512. These inspected traces motivate no selected
checkpoint claim.

## Verification

The independent verifier passed. It regenerated both datasets, reloaded both
checkpoints, replayed 16,384 raw evaluation rows, checked 1,024 optimizer-log
entries, confirmed equal parameter counts, and matched all 16 manifested files.

The local evidence root `artifacts/engram_addressability_g0` has 18 files and
inventory digest
`424a8622090bfe673f4920a0ce1fee329c75077da2c9c3ce09243c36dedc0400`,
computed over sorted `<file SHA256><two spaces><POSIX path><newline>` entries.
The verifier report `artifacts/engram_addressability_g0_verified.json` has SHA256
`583eda696491a6a03897352dc29f5742cc4f20c80e95f0b8156882b6aafd2c20`.

## Allowed repair

One prospective apparatus repair is scientifically motivated: train a single
shared backbone on only the ordinary default task, clone its state into both
arms, freeze every non-memory parameter, and then train the address modules on
the unchanged S0 mixed corpus. This prevents the control backbone from learning
the arbitrary pair map. Pair count, labels, endpoint, optimizer, model width,
table size, seed, and thresholds should remain unchanged. If that successor
also fails S0, stop rather than continue tuning the synthetic task.
