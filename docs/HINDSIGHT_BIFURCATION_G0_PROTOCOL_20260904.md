# Hindsight performative-bifurcation screen

Frozen before inference on 4 September 2026. This is a new mechanism test, not
an extension chosen after seeing its outcomes.

## Claim being tested

Let the constrained policy choose option 1 with probability `p`. A downstream
report names option 0 or 1, and the Hindsight teacher distributions conditioned
on those reports have option-1 logits `l0` and `l1`. If a user simply copies the
deployed answer into the next turn, then the report-1 marginal is also `p`.
Exactly minimizing the report-marginal reverse KL gives

    F(p) = sigmoid((1-p) l0 + p l1).

For a symmetric teacher with logits `-k,+k`, the neutral fixed point changes
from stable to unstable at `k=2`. Above that threshold, two stable, self-confirming
policies can exist and an arbitrarily small initialization difference selects
between them. Freezing the report marginal at its initial value removes this
closed feedback loop.

This differs from the recently rejected mechanism. Direct Brier learning from
the model's own sampled reports contracts the symmetric discrepancy; the
full-distribution reverse-KL distillation map can amplify it. The source of the
difference was established in `HINDSIGHT_SAMPLING_LAW_AUDIT_20260904.md`.

## Prospective assay

- Model: cached `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Thirty-two fresh base prompts across eight unused semantic domains, crossed
  over option order and two wordings. Development and confirmation domains are
  disjoint.
- For each base prompt, measure normalized summed candidate-sequence likelihoods
  under future messages naming each option. This is a constrained two-action
  policy, not free-form generation.
- Enumerate all fixed points of the measured map. At each three-root context,
  start 0.01 on either side of the unstable root and iterate the dynamic map for
  128 exact updates. Compare with a matched control whose feedback marginal is
  frozen at each initial policy.

The screen is positive only if teacher choice accuracy is at least 90% in each
split; at least 8/16 confirmation contexts are bistable; every confirmation
domain contributes at least one; median dynamic endpoint separation is at least
0.5; and the maximum matched fixed-marginal separation is at most 0.1.

## Interpretation boundary

A positive result establishes an executable performative fixed-point mechanism
for the measured conditional teacher. It does not establish human preference
change, welfare loss, free-form behavior, neural-training dynamics, or an ICLR
paper. The exact map assumes complete optimization of each reverse-KL target.
The copying user is deliberately information-free. A useful paper would still
need finite-step shared-parameter training and externally anchored behavioral or
human evidence. Classical performative prediction and pitchfork bifurcations are
major novelty risks.

A negative result parks this Hindsight bifurcation on the present model/template;
we will not alter thresholds or domains to rescue it.

## Verified result

Frozen source commit `9a35a84`. The CPU run completed 96 scored prompts and 192
candidate-sequence forwards in 97.43 seconds with no parameter updates. The
teacher selected the named option 32/32 times in development and 32/32 times in
the untouched confirmation domains; mean target probabilities were .9755 and
.9861 respectively.

The measured map has three fixed points in 9/16 development contexts and 14/16
confirmation contexts. Every confirmation domain contributes at least three of
its four crossings. Around each confirmation unstable point, the median terminal
separation after the two predeclared perturbations is .9935. The largest paired
separation under the fixed-marginal control is .0820. The prospective status is
`BIFURCATION_SCREEN_POSITIVE`.

This is a real positive mechanism result but **not a paper green light**. It is
conditioned on a two-action interface and exact per-round reverse-KL projection;
the next gate is finite-step shared-parameter training with dynamic versus frozen
report marginals. Even if that works, externally anchored behavior is required
to connect self-confirming reports to preference shaping or welfare.

Evidence: `artifacts/hindsight_bifurcation_cpu_20260904_v1`; the adjacent
verification receipt checks all six manifest members, source/case identity,
likelihood arithmetic, row coverage, fixed points, endpoints, and the decision.
It does not replay the 192 neural forwards.
