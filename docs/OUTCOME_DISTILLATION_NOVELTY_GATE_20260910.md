# Outcome-aware distillation: additional novelty gate

Primary-source methods screen, not a reproduction. No GPU work was launched.

| Proposed generic follow-up | Existing primary work | Consequence |
|---|---|---|
| Probe alternative next tokens and use final correctness to adjust targets | [SPOT](https://arxiv.org/html/2608.04419v1), August5 | Already combines position acquisition, verifier-scored student continuations and teacher-anchored target calibration. Generic outcome-aware forks are not a new method. |
| Keep recoverable prefixes; roll back irrecoverable ones | [Counterfactual Recoverability](https://arxiv.org/html/2608.04408v1), August5 | Already compares budget-matched continuation and rollback. Generic recoverability routing is occupied. |
| Handle incorrect consensus differently from ordinary self-distillation | [TTPO](https://arxiv.org/html/2608.27448v1), August27 | Already separates agreeing-rollout distillation from disagreement penalties and studies pseudo-label errors. Wrong-consensus failure alone does not establish novelty. |

Avoid two tempting but unsupported criticisms. SPOT's mathematical improvement
statements are explicitly about estimated continuation values; they do not
guarantee improved unknown true values with finite probes. An elementary noisy-
estimate counterexample would not refute that theorem. The recoverability paper
explicitly identifies its perfect-AUC proxy as branch-derived diagnostic evidence,
not a learned online estimator. Do not present its AUC as an independently
validated deployable classifier and then criticize that invented claim.

The released u-OPSD adapter remains useful infrastructure, but availability does
not differentiate an experiment. No automatic continuation-value or rollback
campaign is admitted. A future proposal must state which of these existing
methods fails under a well-defined condition, why the proposed intervention fixes
it, and how independent outcomes distinguish it from these controls. Existing
PMI target differences and the single faulty-reasoning consensus example do not
meet that requirement.

Remaining questions are hypotheses, not ready paper ideas: whether finite-budget
outcome acquisition can reliably preserve low-probability valid reasoning, and
whether a correction retains its benefit on unseen problem families. Both need
further novelty checks, more than one clean task-level witness, capability and
parser qualification, and a measured compute plan before model runs. Prior
results remain unchanged; no published benchmark result is invalidated here.

## 09:56 follow-up: advantage and teacher-free baselines

[Unmasking On-Policy Distillation](https://arxiv.org/html/2605.10889v1),
Sections2.1-2.5, already defines teacher advantage as the difference between
teacher- and student-weighted continuation success. It renormalizes over a
sufficiently visited token subset and estimates local logit-gradient alignment
using targeted rollouts. AppendixC explicitly acknowledges unvisited-token and
compute limitations; AppendixE.3 labels selective filtering as an oracle analysis.
Thus the contrast in our POSITION_RELIABILITY_AUDIT is not a new diagnostic.
The derivation is at the local-logit level; do not describe it as an independently
validated prediction of a shared-parameter training update. We have not replicated
its measurements or established that its rankings fail after actual training.

[OPSA](https://arxiv.org/html/2608.31046v1), Sections3.2-4.2, already studies
teacher-free negative advantages on selected low-probability tokens and an
entropy-dependent version. Generic teacher removal or entropy-driven tail
suppression therefore also needs this baseline. Its noise label is defined from
advantage sign and final-answer correctness; that operational definition alone
does not establish whether an individual intermediate token is causally harmful.
We have not reproduced its gains or invalidated that study.

Both primary HTML snapshots and hashes are saved in
artifacts/outcome_distillation_screen_20260910/ADVANTAGE_PRIOR_RECEIPTS.json.
The exact remaining-horizon null survives as a measurement warning. It does not
supply the missing novelty or an empirical learning result. Do not admit a large
branch bank for generic teacher-advantage scoring. A distinct experiment would
need an independently motivated failure condition beyond these existing
contrasts, and validation against actual held-out behavior under equal compute.
