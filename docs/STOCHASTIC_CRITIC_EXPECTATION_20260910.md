# Exact stochastic-continuation diagnostic

Classification: **developmental exact expectation**, not neural evidence or a
paper-qualified novelty claim. Script: `scripts/audit_stochastic_critic_expectation.py`.
Artifacts: `artifacts/dvpo_source_audit_20260910/STOCHASTIC_EXPECTATION.json`
and additive-control version `STOCHASTIC_EXPECTATION_V2.json`; both preserved.

## Construction and checks

The root policy selects A/B equiprobably. A fixed subsequent policy chooses
outcomes: A gives 1 with probability .2 and 0 otherwise; B gives .3 with probability
.9 and -1 otherwise. Transitions are deterministic given actions. A final forced
action delivers the reward. Exact state values are .185 at the root, .2/.17 after
the branch, the resolved reward before termination, and 0 at termination. Only
the root policy logit is differentiated. Its true reward gradient is .0075.

All multinomial compositions of batches 1,2,4,8 are enumerated for lambda
0,.5,.95,1. Source hashes are verified before extracting the reviewed released
advantage routine and compatible TRL helpers. No upstream module is imported.
Batch probabilities sum to one. Independent unnormalized recurrence agrees with
.0075*(1-lambda^2); adding the terminal reward gives .0075 for every case.

## What changed

12/16 evaluated configurations have a negative expected root gradient using the
released normalized routine. At batch8/lambda.95 it is -.00862546 versus the true
positive .0075. This establishes a constructed wrong-direction case beyond the
earlier symmetric example, whose direction was correct.

However, removing value whitening retains the reversal (-.00927549 at the same
setting). Even adding terminal reward while retaining advantage whitening gives
-.08601978. Thus the reversal does **not** isolate frozen-value pretraining or
missing terminal rewards. Shared sample-dependent advantage normalization is
sufficient in this construction. The unnormalized terminal-reward control is
correct throughout. Normalized gradient magnitudes have different scales; signs
are the meaningful comparison here.

## Novelty and next decision

Normalization bias is established prior art, including
[The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://api.repository.cam.ac.uk/server/api/core/bitstreams/896d402b-f938-407d-9d95-49631a2f82d8/content),
whose discussion explicitly identifies bias from adaptive normalization. This
turn inspected the search excerpt, not the entire paper; a novelty claim would
require full comparison. The result is not evidence that the released DVPO neural
experiments fail, and its historical TRL version remains unauthenticated.

Do not launch a neural campaign merely to reproduce this generic constructed
bias. A practical contribution requires a materially distinct prediction,
realistic batching, source-faithful critic artifacts and fair existing remedies.
No such qualification has been established. Keep the source finding at diagnostic
scope; continue evaluating other narrowly defined candidates.
