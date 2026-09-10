# Branch viability: remaining-horizon null and required control

**Disposition: developmental measurement audit; no neural finding or admitted
training campaign.** The exact check completed locally in about one second.

Primary source: [PW-OPSD, v1](https://arxiv.org/html/2605.21606v1), Section 3.2
and Appendix A. It selects ambiguous teacher alternatives on correct student
spines, then estimates each child's viability from six ordinary-context
continuations. All children below .40 produce the low-viability label; at least
two children at or above .75 produce the diversity label. Intermediate cases are
excluded. The stated protocol does not describe matched, freshly resampled
unforced continuations at each prefix. This is a bounded protocol observation,
not proof that unpublished controls do not exist. The reported diagnostic has
eight low-viability candidates among 279 binary-labeled candidates, with
problem-cluster uncertainty. Its training gains are separate evidence and have
not been replicated here. The tower-property surrogate identity remains valid;
its mixture identity explicitly does not imply a different gradient objective.
Position weighting is therefore an existing baseline, not a new proposal.

The [official repository](https://github.com/SaFo-Lab/PW-OPSD) tree was pinned at
35303e00ec47fce345d3531322268354fb211e70. Its listed files contain training and
evaluation code, but no separately identifiable branch-diagnostic runner.
Trainer internals were not audited in this check. Source snapshots and receipts
are in `artifacts/position_reliability_audit_20260910/`.

## Exact independent null

Our construction has three interchangeable next-token symbols, each with
probability 1/3 under both teacher and student. After any symbol, h independent
steps each preserve correctness with probability 19/20; an error is absorbing.
Fresh continuation success is therefore (19/20)^h for every forced child and
for an unforced student continuation. A previously correct spine does not change
these independent resampling probabilities. The teacher's incremental expected
success is exactly zero at every prefix, as is its KL from the student.

| Remaining steps | Forced and unforced success | Probability of low-viability label | Probability of diversity label |
|---|---:|---:|---:|
| 1 | .950000 | less than 1e-12 | .996848 |
| 10 | .598737 | .005925 | .135772 |
| 40 | .128512 | .908751 | .000000106 |

These are exact binomial event probabilities, displayed rounded, not Monte Carlo
estimates. The program independently enumerates all 343 triples of success
counts and checks equality with the closed forms. Five horizons pass. See
`scripts/audit_branch_viability_horizon_null.py` and `HORIZON_NULL.json`.

Thus a strong early/late label pattern is possible without any teacher-induced
harm or benefit. This establishes a logical distinction between absolute
continuation difficulty and incremental supervision value. It does **not** show
that this mechanism explains the published data, invalidate absolute viability
as a descriptive measurement, or refute the reported training improvements.
The null is elementary and is not a sufficient paper contribution.

## Consequence for our next experiment

Any outcome-based follow-up must compare forced alternatives with freshly
resampled ordinary student continuations from exactly the same prefix, with
matched decoding and token budgets. A successful historical spine is not that
control. Record termination and parsing separately; an incomplete sample is
not a demonstrated reasoning error. Keep prefix selection independent of these
new outcomes, and cluster repeated branches and positions by problem.

For a target distribution q and student p, the useful local estimand is
sum_a (q(a)-p(a))*V(a), where V(a) is student continuation success after action a.
Absolute V(a) alone does not estimate this contrast. This is a standard advantage
contrast, not a novel method. Top-k-only estimates also need a declared treatment
of excluded probability mass. A positive local contrast still does not establish
improvement after a shared-parameter optimizer update; that requires a separate,
held-out behavioral test. Include uniform, entropy and position baselines before
proposing a new selector, alongside the already-screened outcome-based priors.

No GPU runtime is claimed for that unimplemented follow-up. First qualify
completion/scoring and measure batching throughput on a small development pilot;
then freeze the sample size and budget before its endpoint. Do not spend on a
large branch bank merely to reproduce the exact null. The current paper remains
NO-GO and Hindsight's neural campaign remains parked.
