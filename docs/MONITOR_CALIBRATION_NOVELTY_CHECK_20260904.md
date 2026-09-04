# Monitor calibration: direct collision and sample support

## Proposed question and direct prior work

The proposed distinction between per-call error and whole-session false alarms
is important, but not an unoccupied contribution. The repository had already
flagged Conformal Selective Acting (2605.20270). A new primary-source check finds
an even more direct match:

- [Online Safety Monitoring for LLMs](https://arxiv.org/html/2607.02510v1),
  equations 1–4, explicitly controls the probability of any alarm on a safe
  sequence through trajectory calibration. This covers the proposed basic fix.
- [Conformal Preemption](https://rlj.cs.umass.edu/2026/papers/Paper63.html)
  controls unpreempted failures with a lead-time objective and studies false
  alarms. Recasting the same issue as early intervention is insufficient.
- [Role-Stratified Conformal Risk Control](https://arxiv.org/abs/2607.24343)
  specifically treats structured LLM tool arguments; its abstract describes
  role-level budgets and transfer experiments. Full-method review would be
  needed before any claim about its limitations.

Decision: do not implement or pitch generic trajectory calibration as new.
This is a novelty-screen outcome, not an experimental rejection of a method.

## ScopeJudge support calculation

The pinned data contain 27 trajectories without a majority-positive call, and
only 11 trajectories with unanimous negative votes for every call. If a *fixed*
monitor hypothetically had zero alarms on 27 independently sampled safe episodes,
the exact one-sided 95% binomial upper bound would still be
`1 - 0.05**(1/27) = 0.10501923`. At least 59 independent zero-alarm episodes
are needed for that particular bound to be at most 5%.

This is **not an observed monitor result**, a general lower bound on all possible
certification procedures, or a claim that these episodes are iid. It is a
sample-support warning: 4,897 calls are not 4,897 independent safe trajectories.
Repeated task families and contested safety labels further complicate such a
claim. Conformal expectation guarantees and high-probability risk guarantees
must not be conflated.

Reproducible counts and hypothetical calculation are included in
`scripts/audit_scopejudge_label_structure.py`; fresh receipt
`artifacts/scopejudge_release_20260904_v1/label_structure_v2.json`.

## Research-selection correction

The recent expert agreement interval crossing zero does not establish equivalence
or show that model first-boundary performance is good. That question remains
untested. Parking it is a resource/novelty decision, not a scientific kill.

Likewise, a contribution need not introduce a brand-new causal object *and* a
mitigation to be publishable. A rigorous new empirical finding, useful method,
theoretical result, or benchmark can each be meaningful. Earlier scouting language
requiring all of them was overly restrictive. The proper bar is a clearly
distinguished and consequential contribution with evidence matched to its claim.

The primary paper explicitly suggests combining signals, cost-aware checking,
targeted checks after alarms, and temporal thresholds. These are legitimate
starting points, not automatic novelty: a follow-up needs a concrete advantage
over that paper and existing selective-acting methods. Do not buy a GPU merely
to rediscover the elementary multiple-testing effect.
