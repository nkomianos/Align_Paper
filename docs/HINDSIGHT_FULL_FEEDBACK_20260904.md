# Full-vocabulary feedback learning — prospective staged comparison

Reuse the qualified acquisition dataset/schedule:64 training rows (four phrasings,
eight domains, paired option orders),32 evaluation rows (two other phrasings),
rank8/alpha16 attention LoRA,96 AdamW steps, batch8, lr.0003, clip1, no weight
decay. Each arm resets to the saved **initial**, zero-update acquisition adapter;
it does not inherit the trained oracle's preference knowledge. Preserve the
previous experiments unchanged. This is developmental personalization, not an
independent benchmark or a human-user study.

## Full-vocabulary and invalid-action contract

Freeze base-model full-vocabulary log probabilities for three feedback messages
per training prompt, using the pinned upstream hindsight block:

- Report preference for option A.
- Report preference for option B.
- Request an A/B answer after a different first token.

Let pA/pB be **unconditional** student token probabilities. With probability
1-rho the simulator reveals the true preference; with rho it copies a valid
sampled A/B response, or requests a valid format for any other token. Thus the
three report probabilities are
`rho * [pA, pB, 1-pA-pB] + (1-rho) * one_hot(true option)`.
This avoids silently renormalizing a low-mass A/B policy. It also means rho arms
receive a formatting signal on invalid actions, which must be disclosed rather
than interpreted as a pure preference-only effect. Full-vocabulary output and
A/B mass are reported separately.

Train with the exact report-marginal average of full-vocabulary reverse KL.
Report weights and teacher are stopped; no top-k approximation. This is a
frozen-teacher, one-token, known-simulator marginal objective, **not** a complete
adaptive-teacher SDPO reproduction or same-rollout sampled estimator.

## Ordered arms and stopping rule

First train truthful-feedback KL (rho0). Evaluate only its final checkpoint.
Require the same acquisition criteria: >=90% full-vocabulary argmax accuracy
on32 evaluation rows, >=75% each domain, all A/B masses >=.95. If it fails,
save everything and stop: remaining costly arms are not an informative test of
the hypothesis until truthful hindsight learning works.

If it qualifies, run from the same initial weights and fresh optimizer:

1. Copying KL rho.9.
2. Initial-policy fixed-marginal noisy KL, matched to arm1 at initialization.
3. Copying KL plus anchors (batch-mean KL + mean available-anchor NLL).
4. Anchor-only, identical anchor exposures and96 optimizer steps; zero gradient
   on batches without anchors, with Adam momentum retained.
5. Direct all-label supervision as an execution/positive-control replication.

Anchors retain the previously seeded two domains,16 distinct training phrasings,
192 exposures. Anchor-only and mixed use identical labels. The all-label oracle
uses768 exposures and is not label-budget matched to sparse anchors.

Save full teacher matrices, prompts/token IDs, initial/final adapter and optimizer
states, per-step losses and report probabilities, complete evaluation rows,
source snapshots, counts and hashes. No update while evaluating and no best
checkpoint selection. Report expected true-token probability, full argmax
accuracy, NLL, invalid-output mass, anchored/unanchored and target-label slices.
Compare copying with fixed noise and correction with anchor-only. A/B-only
agreement identities from prior binary experiments do not automatically apply
when invalid actions receive a third feedback type.

No paper greenlight or GPU expansion follows automatically. Success would be
evidence worth evaluating for a broader study, not a guarantee of novelty or
acceptance. A failed clean stage is a teacher-learning limitation in this setup,
not a refutation of the underlying Hindsight hypothesis.
