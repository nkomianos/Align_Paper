# Prior-belief reconstruction: avoid a spurious hindsight result

## A negative control that matters

Suppose latent belief Z does not change, pre-rating P=Z+e0, and post-rating
Q=Z+e1, with independent mean-zero measurement errors. Even an oracle predicting
the exact prior Z has reconstruction error Z-P=-e0. Its error covaries with the
observed change Q-P=e1-e0 by Var(e0), despite zero true change and perfect
latent-state inference. With equal error variances, correlation is 1/sqrt(2).

Executed 200,000 Gaussian synthetic cases: covariance .1600869 versus expected
.16, correlation .707296 versus expected .707107. This is a standard shared-term
measurement artifact, not a new discovery or evidence that PUPPET has that noise.
The simulation is unbounded and is not fitted to the 0–100 human ratings.

Therefore, correlating reconstruction error with measured belief change cannot
be the primary evidence of hindsight contamination. It can give a strong positive
result for a perfect estimator. Larger samples do not repair the estimand.

Implemented paired early/late pre-rating MSE, corresponding post-rating MSE,
and a labeled descriptive alignment statistic. None is itself causal evidence.
The paired MSE contrast is preferable when baseline measurement error is
independent of both predictions: the common noise variance cancels in expectation.
That independence is not automatic if the assistant saw the measured baseline.
User language may also reflect measurement priming; randomization and independent
repeated anchors would provide stronger identification.

Code: `src/interaction_sprint/belief_reconstruction_metrics.py`.
Receipt: `artifacts/belief_reconstruction_metric_null_20260904_v1.json`.
Three tests pass. No human outcomes or transcripts were scored.

## Close primary work to include, not dismiss by theme alone

[DToM-Track](https://arxiv.org/abs/2603.14646) studies recalling prior versus current
beliefs in controlled multi-turn interactions. A generic claim that models forget
previous beliefs is already covered. The possible distinction is inference of
independently measured human beliefs from natural interactions—not merely recall
of supplied fictional states. Its full protocol must be compared before claiming
that distinction is sufficient.

[ExAnte](https://aclanthology.org/2026.eacl-long.72/) studies future-information
leakage under temporal cutoffs. It is related but does not make all historical
user-state inference redundant. [ABBEL](https://bair.berkeley.edu/blog/2026/07/26/abbel/)
trains belief-state summaries with reconstruction-based grading; it is a potential
method comparator, not proof that the proposed human measurement effect exists.

## Decision

Continue design review, but do not announce a paper effect from an error/change
correlation or a generic temporal-recall gap. The needed evidence is a stable,
paired increase in prior-anchor error under well-controlled additional dialogue,
with a useful mitigation or a consequential measurement finding that exceeds
the controlled ToM literature. Dataset use/inclusion questions remain separate
and unresolved; no GPU request is justified yet.
