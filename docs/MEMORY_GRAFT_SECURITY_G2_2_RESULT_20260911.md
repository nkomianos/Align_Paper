# Memory Graft security G2.2 Qwen cross-family result

## Registered decisions

G2.2 validates the row-deletion assay in a second model family and finds
scale-dependent evidence for backbone routing under ordinary fine-tuning.

| Estimand | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---:|---:|
| Surgical target-specific removal, mean [95% CI] | 1.000 [1.000, 1.000], **pass** | 1.000 [1.000, 1.000], **pass** |
| Selected surgical learning rate | 0.001 | 0.01 |
| Frozen-graft attack excess at selected N=16, mean [95% CI] | 0.614 [-0.720, 1.948], **fail** | 0.933 [0.659, 1.206], **pass** |

The surgical result is exact across all six confirmatory runs: intact ASR was
100%, target-row-zero ASR was 0%, and benign-row and random-row controls produced
no competing drop. Near-trigger and untriggered payload rates were zero. This
closes the adapter-specific assay-validity gap at both Qwen sizes.

The routing result is heterogeneous at the smallest prospectively selected dose.
Qwen2.5-0.5B attack excess was 84.28%, 0%, and 99.80% across seeds, so its
three-seed Student-t lower bound fails the 15-point criterion. Qwen2.5-1.5B was
99.22%, 80.57%, and 100%, giving a passing 65.93% lower bound. All original
decisive routing runs had zero near-trigger and untriggered hits. The
difficulty-matched benign continuation was `" qual"`; its clean mean-NLL gap
from the payload was 0.0119 at 0.5B and 0.0204 at 1.5B. It generally was not
learned at N=16, so no matched-learnability claim is licensed.

The correct conclusion is a single-scale positive and a cross-scale negative at
N=16. It establishes that a Qwen backbone can absorb the mapping while the
entire functioning graft is frozen. It does not establish reliable routing at
the smaller Qwen scale under sixteen poison exposures. The development ladder
reached 100% attack excess at N=64 and N=256 for both sizes, motivating a new
fixed-dose reliability experiment rather than retroactively changing G2.2.

## Integrity and verification

The runner completed all 24 registered scientific cells, then failed only in
aggregate assembly because a legacy helper required five values while G2.2
registered three. Before repair, all 84 scientific files were sealed in
inventory digest
`6da08c0a70672b183d03c90ef5a94490a3f4d1c9c5486aaa20fa470a60b03873`.
The independent assembler verified every hash and computed the registered df=2
Student-t intervals without changing raw files. This repair and its verifier
amendment are documented separately.

The amended verifier replayed all twelve decisive optimizer runs and evaluations.
Every registered pass/fail decision reproduced. All 135,168 surgical prediction
rows reproduced bit-for-bit. The 0.5B routing rows also reproduced bit-for-bit.
At 1.5B, BF16 drift changed 2,098 of 12,288 argmax predictions, but the registered
attack-excess values and decision reproduced exactly; the disagreement was in
non-primary outputs. The verified inventory digest is
`6a44469defdf2bf54d56cbb2b20d7b1f8fab9569aafbbe3010e9a5ff09b3331c`.

The scientific run took 2,603 seconds and full replay took 973 seconds. Including
the two preserved pre-scientific failures and prior work, cumulative instance
allocation is approximately 5.08 hours, leaving about 44.92 hours of the original
50-hour budget.
