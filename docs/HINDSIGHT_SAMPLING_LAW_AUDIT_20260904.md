# Hindsight: own-response scoring and fixed-feedback distillation differ

4 September 2026 UTC. **Exact counterexample and exploratory saved-score
analysis; not trained-LM evidence or a paper green light.** This sharpens the
missing learning question. It does not revive the failed anchor prompt.

## Source and interpretation

[Aligning Language Models from User Interactions](https://arxiv.org/html/2603.12273v1)
describes stopped hindsight log-ratio updates and reverse-KL distillation.
Section 3 and appendix B state an expected-gradient equivalence. Applying the
fixed-feedback sampling identity to the original response and its own feedback
needs care. The [pinned released updater](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/online_sdpo_updater.py)
implements `_simple_signal_loss` on supplied response tokens and
`_full_distillation_loss` over student top-k probabilities, with optional tail.
The offline trainer likewise scores supplied logged completions.

The inspected arXiv record offered v1. A separate OpenReview PDF appeared in
search but direct access hit a browser challenge. No claim is made about
whether later versions clarify this issue, or that published empirical results
are false. This audit concerns the explicitly pinned source and sampling laws.

## Exact distinction

At a fixed context draw A from policy p, then O from K(o|A). Let
M(o)=sum_a p(a)K(o|a), fix positive teacher q_o(a), and write s(a)=grad log p(a)
and d(a,o)=log q_o(a)-log p(a). Directions below are negative loss gradients.
Teacher and data-sampling weights are detached in both calculations.

    g_own  = sum_{a,o} p(a) K(o|a) s(a) d(a,o)
    g_full = sum_{a,o} p(a) M(o)   s(a) d(a,o)
    difference = sum_{a,o} p(a)[K(o|a)-M(o)]s(a)d(a,o)

The first scores the action that caused the feedback. The second computes
full reverse KL for fixed feedback and averages over the collected feedback
marginal. A fresh independent action drawn after feedback is fixed estimates
the second. The original action generally has a different conditional law
after its feedback is observed. Equality holds under action-independent
feedback, or special cancellation of the displayed difference.

This is not merely missing a derivative through the teacher. Neither expression
includes the upstream derivative of a changing interaction distribution; that
would define a third target. Neither is necessarily a welfare gradient.

### Sign reversal

Use K(1|0)=0.1, K(1|1)=0.9 and Bayesian teacher
q_o(a)=p(a)K(o|a)/M(o), refreshed then detached. At p=P(A=1)=0.1:

| Calculation | Own-response ascent | Full reverse-KL ascent |
| --- | ---: | ---: |
| Exact enumeration | +0.1091770192 | -0.1265601357 |
| Released loss functions on mocked logits, autograd | +0.1091770192 | -0.1265601357 |
| Frozen-target finite difference | different target | -0.1265601357 |

The own-response direction is our earlier mutual-information derivative:

    p(1-p)[KL(K_1 || M)-KL(K_0 || M)]

The full-KL direction for this symmetric channel is:

    p(1-p) * 0.8 * (2p-1) * log(9)

The former restores balance; the latter amplifies imbalance. The earlier
sampled-update calculation remains correct, but must not be generalized to
both losses. Included 1,000-step exact-logit trajectories (step size 0.2)
illustrate these dynamics, not human preference changes or LLM training.

Weighting each own-response coefficient by M(o)/K(o|a) restores g_full on
positive support. This is standard importance weighting with an oracle channel,
not a new deployable correction. Here it restores the imbalance-amplifying
direction: unbiasedness is not desirability.

## Already measured LM scores

Reverified the old 512-forward Hindsight archive and reused saved Qwen3-4B
probabilities. Identical base/follow-up prompts have matching probabilities;
20 unique prompts cover four surfaces and two wordings. For each of eight
surface/wording contexts, use the A/B-normalized base distribution and the two
measured feedback-conditioned distributions. No Bayesian-teacher assumption
is used in this second calculation.

Impose a channel that copies the action with probability c and otherwise
emits a fair-coin feedback choice:

| Imposed c | Contexts with opposite nonzero A/B-logit directions |
| --- | ---: |
| 0 | 0/8 |
| 0.25 | 3/8 |
| 0.50 | 7/8 |
| 0.75 | 7/8 |
| 0.80 | 7/8 |
| 0.90 | 7/8 |

These are post-hoc calculations under analyst-designed feedback laws, not
estimated user-channel frequencies. The contexts are not independent model
replications. Restricted A/B-logit directions need not match shared neural
parameter gradients. The prompts are our earlier simplified hindsight template,
not the exact upstream template. No new inference, learning or welfare result
is claimed. Original decisions and evidence remain untouched.

## Novelty and next test

[Privileged Likelihood Is Not Automatically Value](https://arxiv.org/html/2608.09263v1),
sections 2.4, 3.3 and appendix A.4, already distinguishes own-rollout feedback,
cross-fitting, and partial versus upstream gradients. It also compares how
losses use teacher scores. Generic endogenous feedback or gradient mismatch
is therefore not a new claim. The narrower candidate concerns exchanging
purportedly equivalent estimators under the actual feedback law and their
online fixed points. The identity alone is elementary; priority and sufficient
contribution remain unestablished. Removing dependence does not establish a
useful learning signal.

Next: design a small learning-level diagnostic comparing own-response scoring,
full reverse KL, and independent-resampling estimation of that same full-KL
objective. Include action-independent feedback as a null and known-channel
weighting only as an oracle. Separate objective fidelity from task utility.
Freeze fresh tasks, feedback rules, exact prompts, vocabulary scope and
optimizer budgets before launch. Verify parameter gradients and no-op controls;
preserve all checkpoints. Do not choose feedback strength after observing the
largest gap, or interpret the 7/8 calculation as a paper pass. No GPU launch or
running queue was created during this audit.

## Reproduction

`src/interaction_sprint/feedback_gradient_audit.py`; upstream commit
`3b17d2a67bd2565b9fbda495fd16a485406aa954`, canonical loss-file SHA-256
`6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d`.
Only three inspected loss methods are extracted; model forwards are mocked.
V=top-k=2, no tail, no signal clipping, one response token. This does not run
the full upstream model/trainer or reproduce its experiment.

Reports: `artifacts/feedback_gradient_audit_v1.json` (finite model) and
`artifacts/feedback_gradient_audit_v2.json` (also saved-score analysis). Tests
cover analytic gradients, finite differences, independent-feedback null,
oracle weighting, dynamics, support failures and released loss execution.
