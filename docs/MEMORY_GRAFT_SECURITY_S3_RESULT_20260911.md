# Memory Graft security S3 result: backbone-component localization

## Registered question and validity

S3 used the five sealed S1 training seeds at Pythia-410M and Pythia-1.4B to
intervene on poisoned and clean checkpoints. Every intervention restored the
complete graft to the seed-matched clean state. The functional partition
(embedding/head, attention, MLP, normalization) and depth partition (four
six-layer bands) were fixed before checkpoint loading.

The endpoint controls passed. Clean and poisoned no-ops reproduced exactly,
and complete-backbone transplantation/restoration recovered the S2 contrast:
the mean effect was 0.9965 (95% CI [0.9912, 1.0018]) at 410M and 0.9949
([0.9862, 1.0036]) at 1.4B. The source-manifest audit passed for all 23 files.
The independent full replay reproduced every registered decision and every raw
prediction ID across all ten model/seed files.

## Registered results

The table reports poisoned-ASR drop after restoring a group to its clean value
(necessity) and clean-adjusted ASR after transplanting that poisoned group into
the clean backbone (sufficiency). Intervals are 95% Student-t intervals over
the five training seeds. PASS means the lower endpoint exceeds the registered
0.15 meaningful-effect threshold.

| Model | Group | Necessity mean [95% CI] | Decision | Sufficiency mean [95% CI] | Decision |
|---|---|---:|---|---:|---|
| Pythia-410M | all MLP | 0.9965 [0.9912, 1.0018] | PASS | 0.3680 [-0.1479, 0.8839] | FAIL |
| Pythia-410M | layers 0--5 | 0.9596 [0.9161, 1.0030] | PASS | 0.0000 [0, 0] | FAIL |
| Pythia-410M | all attention | 0.2619 [-0.1397, 0.6636] | FAIL | 0.0000 [0, 0] | FAIL |
| Pythia-1.4B | all MLP | 0.9949 [0.9862, 1.0036] | PASS | 0.4814 [-0.0233, 0.9862] | FAIL |
| Pythia-1.4B | layers 0--5 | 0.9252 [0.8464, 1.0040] | PASS | 0.0000 [0, 0] | FAIL |
| Pythia-1.4B | all attention | 0.1570 [-0.0557, 0.3698] | FAIL | 0.0000 [0, 0] | FAIL |

Embedding/head, normalization, and each later depth band failed the registered
necessity and sufficiency criteria in both sizes. These failures are failures
to establish an effect at the registered scale; they are not equivalence
claims. Attention and later-band necessity were notably variable across
seeds, so their point estimates must not be described as absence.

## Conclusion and boundary

The ordinary learned mapping is causally dependent on both the collection of
MLP updates and the early six-layer band in both sizes. No registered coarse
component is reliably sufficient by itself. This is evidence for checkpoint
co-adaptation and rules out a self-contained graft or a single sufficient
coarse backbone group under this recipe. Because the two passing groups
overlap but were tested separately, S3 does **not** identify early-layer MLPs as
the locus. That intersection requires a separately frozen intervention.

## Provenance

- Runner wall time: 563.78 seconds on one NVIDIA RTX PRO 6000.
- Source output manifest: 23 files; SHA-256
  `21ac7eecf3f5ad47751e2c7cc149294ca57ff481f97df17623fa525770e9afbd`.
- Frozen implementation commit:
  `cf03b1b8b21394e81edb43c89bcfba0ebeb46892`.
- Replay verifier SHA-256:
  `146a5d8384681ad40a492aecf90cc9e69429cb1af120bb23d5f9918aea7caaae`.
- Replay outcome: PASS; zero prediction-ID disagreements.

