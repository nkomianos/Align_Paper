# Memory Graft security S4 result: early-MLP intersection

## Registered question

S4 prospectively selected the intersection of the two coarse groups that passed
S3 necessity: MLP parameters in layers 0--5. It reused all five sealed S1 seeds
at Pythia-410M and Pythia-1.4B, restored the complete graft to clean before each
intervention, and performed no training or checkpoint selection. The primary
necessity and secondary sufficiency estimands were separate. Each used the
pre-registered 0.15 meaningful-effect threshold.

## Result

All endpoint controls passed and both no-ops reproduced exactly.

| Model | Early-MLP necessity mean [95% CI] | Decision | Sufficiency mean [95% CI] | Decision |
|---|---:|---|---:|---|
| Pythia-410M | 0.7104 [0.5056, 0.9151] | PASS | 0.0000 [0, 0] | FAIL |
| Pythia-1.4B | 0.4828 [0.1349, 0.8307] | FAIL | 0.0000 [0, 0] | FAIL |

The 1.4B lower endpoint misses the fixed 0.15 threshold by 0.0151. The threshold
is not relaxed and the result is not promoted from suggestive to positive.
Early-layer MLP updates form a necessary causal locus at 410M under the tested
recipe. At 1.4B, the direction and mean are consistent with the same mechanism,
but seed variability prevents the thresholded claim. The poisoned early-MLP
weights alone are insufficient at both sizes, supporting co-adaptation with
other poisoned backbone parameters.

## Verification and provenance

- Source runner wall time: 270.27 seconds on one NVIDIA RTX PRO 6000.
- Frozen implementation/preregistration commit:
  `43646d0b7d4d8acf6a4d2627c1ea168026b097d5`.
- Freeze receipt commit: `ea5b5c2`.
- Output manifest: 23 files, validated locally without mismatch.
- Full replay report SHA-256:
  `3cc282887f613baee9d6281d90b58a73ab94937d8b2d7e55f61ecfc889ceb9af`.
- Full replay result: PASS; both registered model decisions reproduced and all
  prediction-ID disagreement counts were zero.
