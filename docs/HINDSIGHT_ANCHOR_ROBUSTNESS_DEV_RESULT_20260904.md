# Delayed-anchor robustness DEV result

## Decision

`DEV_ROBUSTNESS_LAW_QUALIFIED`. All six frozen criteria pass, and deterministic
read-only replay verifies the 320-cell result. This strengthens the case for a
neural delayed-anchor test, but it is not a paper green light or evidence about
human feedback.

Evidence root:
`artifacts/hindsight_anchor_robustness_dev_20260904_v1`. Manifest SHA-256:
`180130ed3713c186cc4d4b0eff970ece6e996527a53dd74427f025a9d84560ba`.

## Main result

With clean random anchors, mean initial-preference regret across the 13
immediate-ranking-reversal settings is:

| Method | Mean regret |
| --- | ---: |
| Raw immediate feedback | .345673 |
| Equal-anchor Horvitz--Thompson | .006765 |
| Augmented IPW residual | **.004177** |

The augmented estimator reduces regret 38.3% relative to the equal-anchor
comparator. This is the same sample-efficiency mechanism seen in the first
matched audit, now expressed in a setting with missing delayed outcomes.

When observation strongly favors positive immediate reports, naive estimators
are biased:

| Method | Mean absolute bias at maximum selection |
| --- | ---: |
| Unweighted anchor-only | .120178 |
| Unweighted augmented | .020851 |
| IPW anchor-only | .001628 |
| IPW augmented | **.001243** |

Across every clean selection condition, maximum absolute augmented bias is
`.106438` without weighting and `.0056125` with known-propensity weighting.
Thus post-response participation is not innocuous, but random invitations or a
valid observation model can correct selection that depends on the observed
immediate report.

The qualification law is elementary and exact. For positive known
`e(A,O)=P(S=1|A,O)` and `S independent of B conditional on A,O`, within each
randomized action:

`E[O] + E[S/e(A,O) * (B-O)] = E[B]`.

This is a Horvitz--Thompson difference estimator. It is not claimed as a new
statistical estimator; its role is to make the interaction-learning design
valid under selective follow-up.

## Hard limitation: a measured wrong target remains wrong

Contamination mixes the initial-preference agreement `T` with immediate report
agreement `O`:

`B_rho = (1-rho) T + rho O` in expectation.

Weighting correctly identifies `E[B_rho]`; it cannot infer `E[T]` without an
additional measurement model. In 176 informative rank-preserving cells, the
IPW augmented policy matches the initial-preference optimum 96.55% of the time.
In 44 cells where contamination materially reverses the anchor target, it still
matches that original optimum only 2.42% of the time. That is the intended
failure diagnostic: estimator consistency cannot rescue estimand validity.

For a baseline action-value difference `d_T>0` and immediate difference
`d_O<0`, the analytic flip occurs at

`rho* = d_T / (d_T-d_O)`.

For the neural gate's main population (`p=.75,c0=1,c1=0`), this boundary is
`rho*=2/3`. The synthetic neural G0 uses uncontaminated, randomized anchors, so
it tests the method on the identified side of this boundary. Later human-facing
work must justify anchor timing, wording, nonresponse assumptions and
sensitivity to contamination rather than treating any delayed reply as ground
truth.

## PI interpretation

This result improves the paper architecture:

1. observational next-turn logs cannot separate expression from transition;
2. sparse delayed measurements identify a declared longitudinal target;
3. abundant immediate logs reduce variance through a residual estimator;
4. known response propensities extend the method to selective follow-up; and
5. anchor contamination creates a calculable partial-identification boundary.

The remaining acceptance bottleneck is empirical, not another tabular sweep.
The frozen Qwen3.5-9B neural G0 must show that this correction works through the
SDPO teacher and shared parameters and beats equal-anchor baselines. A pass then
requires released-loss, full-sequence and multi-seed replication plus evidence
that a defensible anchor can be measured in actual interactions.

