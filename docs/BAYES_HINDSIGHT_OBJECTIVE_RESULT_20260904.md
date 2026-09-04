# Exact objective audit: opposite dynamics under different expectations

## Finding

For a binary symmetric response channel with crossover e, policy probability p,
and a stopped exact Bayesian posterior teacher, averaging full reverse-KL
distillation over the policy's marginal feedback distribution gives logit descent
direction

`p(1-p)(2p-1)(1-2e) log((1-e)/e)`.

For 0<e<1/2, p=1/2 is repelling under infinitesimal updates. In contrast, the
previous own-action sampled log-ratio direction restores balance. At p=.1,e=.1,
the two directions are -0.126560 and +0.109177, respectively. Expected forward-KL
distillation has zero direction by the posterior averaging identity. The
uninformative channel has zero direction for all three rules.

These expectations are different: full-distribution distillation considers all
candidate actions for a sampled observation; the own-action score uses the action
that actually generated the observation. The existing `theory.py` computation
remains correct for its stated rule. It does not establish stability of full
reverse-KL distillation. No previous evidence was changed.

## Verification and failed expectation

The first new test incorrectly expected both directions to be positive at p=.1.
It failed. Rather than changing implementation signs, derived the symmetric
closed form above and independently finite-differenced the stopped-teacher KL
loss with both teacher and observation distribution fixed. Both agree with the
implemented negative reverse-KL direction. The test now encodes that analytically
verified result. Four tests pass, including 100 random-channel forward-KL checks.

Raw 15-cell output is retained in
`artifacts/bayes_hindsight_objective_audit_20260904_v1/audit.json`.
Implementation: `src/interaction_sprint/bayes_hindsight_objective_audit.py`.

## Scope and next test

This is an elementary finite-model finding, not yet a novel theorem or an SDPO
failure. Real teachers are not exact Bayesian posteriors; token-level top-k
distillation, sampled prefixes, adaptation and repeated-user states differ.
Nor does imbalance alone prove welfare loss. The uninformative/static channel
cannot demonstrate useful adaptation under this teacher assumption.

Next: audit the actual released full-distillation expectation against this model,
and check literature on mode-seeking distillation before claiming novelty. A
useful result would link the objective difference to an observed learning effect
while retaining benign correction capability. Do not launch a GPU replication
until that mapping and a nontrivial correction baseline are specified.

## Released loss mapping executed

`scripts/probe_released_sdpo_binary_loss.py` extracts the unmodified
`_full_distillation_loss` and `_renorm_topk` methods from the pinned upstream file
(SHA-256 `6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d`).
Controlled two-token logits replace model forwards. With top-k=2 and no tail,
autograd through the released method agrees with the analytic marginal-feedback
direction to absolute tolerance 1e-12 at all five policy probabilities.
Torch 2.11.0+cpu; receipt `artifacts/released_sdpo_binary_probe_20260904_v1/probe.json`.
This tests the loss operator, not default top-20/tail approximation, tokenization,
the Bayesian teacher assumption, or a real LM's learning dynamics.

## Closest prior work found

Nicolicioiu, Pezeshki and Courville's June 2026 paper already establishes
probability-gap amplification and reduced diversity under sampled-demonstration
self-distillation, with theory and experiments:
https://arxiv.org/abs/2606.26091 . Its stated scope excludes richer environmental
feedback. Consequently, our binary example is not enough for a new paper.
The possible extension is how action-dependent feedback changes the dynamics
and a specified user objective, with a correction that still learns useful
feedback. The broader August review is useful context, not primary experimental
evidence: https://arxiv.org/html/2608.25936v1 .
