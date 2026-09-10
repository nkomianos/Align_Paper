# Benign mechanistic direction reassessment

The generic steering/answer-encoding audit is already occupied. Gao, Peng,
Wakamiya and Aramaki, arXiv2608.22985v1 (24August2026), explicitly freeze
interventions while changing answer encodings and distinguish semantic labels,
extraction identifiers and row positions. Their abstract reports task-dependent
effects and differences between MCQ and open-ended evaluation. Official abstract,
author list and full-text structure inspected:
https://arxiv.org/abs/2608.22985 and https://arxiv.org/html/2608.22985v1.
No raw reproduction or full independent audit of their results is claimed.
Do not advertise re-encoding steering targets as our new method.

An existing, still unimplemented candidate is more distinct:
CANDIDATE_RECIPE_INVARIANT_CAUSAL_MECHANISMS.md proposes selecting interventions
on training recipes A/B and predicting causal transfer to a held-out recipe C.
Its historical lack-of-authorization wording predates the user's current broad
authorization; the actual blocker is scientific/implementation readiness, not
a need to ask permission again. Only benign tasks are within current scope.

The Model Organism Lottery, arXiv2607.01033v1, already studies54 small model
organisms across seven training techniques and reports substantial methodology
dependence after behavioral-strength controls. Rechecked official abstract and
full-text structure at https://arxiv.org/abs/2607.01033 and
https://arxiv.org/html/2607.01033v1. Generic recipe sensitivity, SFT-versus-DPO
differences or weak interpretability in integrated training are prior results.
The proposed held-out selection rule remains a hypothesis requiring deeper
method/source comparison; it is not novelty-cleared by this abstract check.

Before a training launch, the following concrete questions must be resolved:

1. A benign behavior must have independently verifiable semantic expression,
   ordinary-task capability controls and adequate headroom. A nonce answer-token
   preference alone would be an output-encoding organism, not a broad mechanism.
2. Define three materially different training recipes and distinguish genuine
   integrated post-training from merely mixing unrelated examples into late SFT.
3. Freeze a common candidate-direction representation, equal intervention norms
   and a rule selected exclusively on A/B. Compare pooled means, single-recipe
   selection and matched semantic controls; beating random directions is weak.
4. Match expression on calibration data before opening recipe-C evaluation.
   Calibrating on C's target outcomes would destroy the claimed held-out test.
5. Include task/seed replication; prompts within one trained model do not supply
   independent evidence across recipes. Estimate actual training/inference cost
   with an apparatus preflight before admitting the full run.

No new code, training job, causal result or paper qualification is claimed.
This candidate merits a bounded readiness audit as a previously discussed benign
idea, rather than another run of stopped temporal or feedback screens. If a
useful held-out selection method cannot be distinguished from strong pooling
baselines, stop before training. The broader submission goal remains unchanged.
