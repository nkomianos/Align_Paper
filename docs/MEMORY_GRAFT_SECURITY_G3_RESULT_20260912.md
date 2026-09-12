# G3 optimizer-routing result

## Status

**Verified at Pythia-410M; verification failure at Pythia-1.4B.** The source
manifest contains 86 files and validates locally without mismatch. The frozen
full replay is bit-exact at 410M and reproduces its registered decisions. At
1.4B, BF16 numerical variation changes the developmental profile selected by
the registered maximum rule, so the cross-scale G3 claim does not survive.

## Source result

Both sizes selected joint AdamW with table learning rate 0.1 in the source run.
At 410M, whole-table necessity is 0.8783 (95% Student-t interval
[0.5418, 1.2148]) and whole-table sufficiency is 0.6363 [0.1683, 1.1043].
Both lower endpoints exceed the registered 0.15 effect. Outside-table
sufficiency is 0.00020 [-0.00035, 0.00074]. Thus aggressive table optimization
moves the learned behavior from a sufficient backbone copy to table-dependent
state at this size.

The nominal final trigger rows do not form a reliable item boundary. Their
necessity is 0.4787 [-0.1449, 1.1024] and sufficiency is 0.3006
[-0.1062, 0.7073]. Individual target-row deletion effects range from zero to
one. The complete table can be necessary even when the nominal rows are not,
which motivates a separately prospective row-footprint localization study.

At 1.4B, the source run's whole-table necessity lower endpoint is 0.0594 and
all registered row and outside-table decisions fail. These are preserved source
measurements, not verified positive conclusions.

## Replay failure and licensed interpretation

The verifier reproduces all 410M prediction IDs exactly. At 1.4B, it selects
AdamW table learning rate 0.001 rather than 0.1 because none of the profiles
passes development eligibility and small BF16 changes alter which profile has
the maximum sub-threshold row-sufficiency value. Consequently the 1.4B decision
reproduction flag is false and the complete verifier status is false.

G3 therefore licenses one scale-specific conclusion: at 410M, optimizer
allocation causally changes the storage component, but does not make the nominal
query rows a reliable deletion boundary. It does not license cross-scale
optimizer-routing claims. A fixed-profile successor must remove the unstable
developmental argmax before any 1.4B claim.

## Integrity and compute

- Source manifest SHA-256:
  `d14f42de1d46946aee3f02f985fd5247dc8cf44a2dce645d832b0b5136a51881`
- Verification report SHA-256:
  `9a5b79ab35d87bba62536d66be262dea5448138e58984c59188032b24c09a8ff`
- Source wall time: 2,293.06 seconds.
- Full replay wall time: 2,309.08 seconds.
- Combined measured GPU time: approximately 1.28 hours.
- Approximate cumulative program use: 7.51 of 50 GPU-hours.

