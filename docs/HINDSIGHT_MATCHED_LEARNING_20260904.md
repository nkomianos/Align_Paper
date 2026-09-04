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
