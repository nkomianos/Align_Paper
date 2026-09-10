# Frozen value implementation audit: a concrete unresolved comparison

Status: developmental mathematical/source audit. No neural reproduction or
paper qualification. This provides a specific next investigation for the earlier
portable-value proposal, rather than repeating the invalid MATH assay.

## Primary-source scope

[DVPO v2](https://arxiv.org/html/2502.16944v2), sections 3.1–3.4, describes a
trajectory-conditioned critic and direct value-based advantages. Therefore,
policy-dependent values alone cannot refute its proposal. Its equivalence lemma
assumes both estimators approximate the target policy's values; it does not
establish that arbitrary frozen critics retain that accuracy as policies change.
The empirical tables have not been reproduced here.

The [official release](https://github.com/microsoft/DKI_LLM/tree/c090b7d57e3996c8900e86fecd16085b27e07f29/dvpo)
was pinned at c090b7d57e3996c8900e86fecd16085b27e07f29. Nine source files were
downloaded and verified against Git blob identities and SHA256 receipts.
No dependency installer or training entry point was executed.

## Observed implementation, not a claim about historical runs

In policyv1_trainer.py, prepare_model_inputs concatenates query and response.
batched_forward_pass passes the same inputs to the actor and critic, whitens
critic outputs before masking, and drops their final position. compute_rewards
returns KL penalties; compute_advantages applies the GAE recurrence with default
gamma=1, lambda=.95. The inspected instruction entry point constructs queries
from the current question, without an additional policy demonstration.
This source path needs reconciliation with the paper's trajectory conditioning
and direct-value description. Unreleased training configurations may differ.

## Independent arithmetic result

For fixed supplied state values, zero external reward, and zero terminal next
value, the recurrence is

    delta_t = gamma V_(t+1) - V_t
    A_t = delta_t + gamma lambda A_(t+1).

At lambda=1 the finite sum telescopes exactly to A_t=-V_t. For a genuinely
state-only baseline, its expected unclipped on-policy score gradient is zero.
This conclusion does not automatically survive sample-dependent normalization,
minibatch whitening, parameter-dependent baselines, clipping or reused samples.

For two length-two paths with common V_root=1/2 and next values 1 versus 0,
the expected root Bernoulli-logit gradient at probability1/2 is (1-lambda)/4.
It is1/80 at lambda=.95 and zero at1. Thus a bootstrap-dependent signal can
exist below1 even without terminal rewards. This is standard GAE algebra,
not a new theorem, and does not measure trained language-model performance.

scripts/audit_frozen_value_recurrence.py checks64 exact rational telescoping
cases plus four two-path examples. Source receipts and results are in
artifacts/dvpo_source_audit_20260910. All checks passed.

## Smallest decisive next step

Before GPU use, isolate and execute the released advantage routine on controlled
tensors, preserving its actual normalization and masking. Compare against direct
state-action values, against a reward-bearing GAE positive control, and against
the analytic unnormalized recurrence. Include varying padding, prompt lengths,
response lengths and minibatch composition. Distinguish state from action-value
indexing explicitly. Estimated CPU work is minutes once the isolated harness is
implemented; no honest neural runtime estimate is available yet.

If source execution disagrees with this reading, correct this note and stop the
critique. If the difference survives, freeze a benign neural gradient experiment
to measure whether it matters before contemplating any training campaign. The
scientific endpoint must be useful reward-directed learning, not merely a code
discrepancy. Full-policy replication remains conditional and unrun. Generic
frozen-critic bias and potential-shaping identities are established prior art;
neither alone justifies an ICLR submission.
