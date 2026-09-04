# Matched preference learning — frozen before training

This is actual parameter learning after teacher qualification. It is a small
developmental test, not a reproduction of full-vocabulary, adaptive-teacher SDPO.

## Fixed design

- Cached Qwen3-0.6B, same pinned revision; CPU float32/eager, four threads.
- Attention LoRA rank4/alpha8, 24 AdamW updates per arm, lr .0003, no weight decay,
  gradient clip1; fresh identical optimizer and adapter initialization per arm.
- Six passes through 16 preference contexts (eight domains, both option orders).
  One semantic preferred option per domain assigned by seed90411031, without
  choosing labels from model scores. A/B labels exactly balanced.
- Known user preferences absent from the student prompt. Explicit teacher
  feedback reveals an option; use the frozen base model's published-template
  probabilities from the prior 128-forward experiment. Teacher does not refresh.
- Same 24 batches of four; exactly one anchor context per batch. Four unique
  anchored contexts (two semantic domains), 24 anchor exposures, in both anchor
  arms. Mixed arm uses mean categorical KL plus mean anchor NLL, weight1.
- Evaluate both training wording and a fixed new wording, on the same semantic
  preferences. This is personalization/paraphrase transfer, not generalization
  to unseen people, preferences or domains. All cases retained.

## Arms and the important new control

1. Truthful-feedback full categorical reverse KL.
2. Response-dependent copying feedback: report true preference with probability
   .1 and copy the current sampled assistant option with probability .9.
3. **Fixed-marginal control:** freeze that feedback distribution at the initial
   policy. Starts with the same feedback distribution as arm2, but does not change
   as the policy changes. Separates a changing feedback loop from static noise.
4. Arm2 plus supervised preference anchors.
5. Anchor-only supervision, same unique labels/exposures as arm4.
6. Direct truthful supervision of all contexts: strong positive control using
   exact labels from the truthful feedback. Not label-budget matched to sparse
   anchors or corrupted-feedback learners.

The known simulator permits exact averaging over actions and reports. Sampling
weights are stopped in differentiation, as are teacher probabilities. Raw KL
arms do not receive true preference as supervised labels, although the simulator
uses it to compute feedback probabilities. This is an oracle-expectation study,
not a learner estimating the feedback law from real logs.

## Interpretation

Report accuracy and NLL in both views, by anchored/unanchored domain and option
order, alongside no adaptation. No best-checkpoint or best-domain selection.
Compare copying to both truthful and fixed-marginal controls. Compare the mixed
correction to anchor-only, not only to the uncorrected learner.

In this particular fixed-copy-strength model, reported agreement equals
`rho + (1-rho) * true-preference utility`. Consequently it cannot increase while
that utility decreases within a fixed-rho arm. Do not manufacture that desired
paper signature: this test measures feedback-loop learning and correction only,
not persistent preference shaping or the full initial report's claim.

A null or weak result in this tiny frozen-teacher model is not a universal
rejection. A positive result is not a paper greenlight without robustness,
larger-model/full-method validation and novelty beyond known feedback dynamics.
Preserve all adapters, optimizer states, case mappings, scores and failed runs.

## Independent implementation review (without viewing outputs)

A separate reviewer checked the loss, fixed-marginal control, adapter/optimizer
resets and anchor scaling and found no fatal flaw within the stated scope.
The full categorical KL's inner action is averaged independently of the action
that generated feedback; it must not be relabeled as a same-rollout sampled
SDPO estimator. This distinction is especially important with action-dependent
feedback. The agreement identity above uses expected sampled utility,
`mean pi(true preference)`, **not argmax accuracy**. Analysis will report both.

The verifier adds anchored/unanchored and A/B-target breakdowns from saved rows.
The code saves final optimizer states for completed arms; its exception handler
saves the current adapter but not the current optimizer. This provenance limit
is recorded prospectively and must not be represented as full failure-state
preservation if an arm fails.

## Completed and independently checked

Frozen training commit `d0fea1b`. Successful CPU completion in **361.86 seconds**:
728 forwards, 504 backwards, **144 actual parameter updates** (24 per arm).
All six final adapters and optimizer states plus the initial adapter are saved.
The read-only verifier checks hashes, case/schedule identity, teacher identity,
sampling marginals, recorded loss arithmetic, equal first copying/fixed-control
batch probabilities and gradient norm, outcome arithmetic and saved states.
It does not replay neural backwards or optimizer steps. Maximum independent
loss-recomputation error is 8.88e-16. Initial probabilities match the preceding
base-model scores within 3.47e-18.

Changed-wording results (16 paired option-order cases, eight domains):

| Method | Correct | NLL (lower better) | Mean true-option probability |
| --- | ---: | ---: | ---: |
| No adaptation | 8/16 | 3.384 | .5019 |
| Truthful feedback KL | 8/16 | 1.480 | .5113 |
| Response-dependent copying KL | 8/16 | 3.633 | .4993 |
| Fixed-marginal noisy KL | 8/16 | 2.552 | .5011 |
| Copying + anchors | 9/16 | 3.059 | .5728 |
| Anchors only | 8/16 | 3.225 | .5088 |
| Direct truthful supervision | 7/16 | .703 | .5074 |

Training-wording accuracy/NLL: baseline8/16,4.286; truthful10/16,1.286;
copying7/16,6.559; fixed8/16,2.916; mixed10/16,3.458;
anchors10/16,2.180; direct10/16,.624. All slices, including target-label and
anchored/unanchored breakdowns, are in the verified receipt.

### Interpretation that survives the controls

There is a descriptive feedback-loop effect in this run: copying makes NLL
worse than the initially matched fixed-noise control, particularly on training
wording. On changed wording the expected sampled utility difference is only
about **−0.18 percentage points**, not a large performance collapse. Most raw
arms still always choose A there. Do not translate a large NLL difference into
a claim of a large welfare effect or hide the accuracy/utility results.

Anchors improve the mixed model on changed wording, but the extra correct case
is anchored: both mixed and anchor-only remain **6/12 on unanchored cases**.
There is no demonstrated unanchored accuracy-transfer benefit. Mean target
probability improves more broadly only if separately established from the rows;
aggregate probability alone is not such a demonstration.

The all-label positive control achieves only10/16 on training and7/16 on changed
wording after this fixed short schedule. Its improved NLL largely removes
overconfident errors rather than establishing robust preference acquisition.
Its minimum A/B mass also drops to .894 (train)/.920 (changed wording), so the
categorical metrics increasingly condition on a restricted output subset.
This is a weak training/control regime, not a clean refutation of the Hindsight
thesis. A larger causal interpretation or a successful corrective learner is not
supported. Keep all results; do not silently extend these completed arms or
select another checkpoint to change their decision.

The next necessary step is a separate prospective acquisition calibration with
adequate updates and a strong no-feedback-label learning control. Only once it
can learn the intended preference task does comparing noisy feedback become
decisive. Do not spend on a larger multi-arm sweep just because this one is weak.

Root: `artifacts/hindsight_matched_learning_cpu_20260904_v1`.
Receipt: `artifacts/hindsight_matched_learning_cpu_20260904_v1_verified.json`.
No active CPU/GPU process remains for this experiment.
