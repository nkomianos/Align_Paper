# Hindsight filter: measurement and observation-cost stress test

## Design

CPU-only developmental simulation, not an LM result. Extends the previous grid
with q=.45 and .55 (six preference means total), retaining three copying rates
per action and 16 anchors. Each of 54 cells has 2,000 repetitions. There are
500 feedback reports per randomized action. The pre-treatment anchors remain
truthful iid Bernoulli samples; this is still a strong assumption.

The stress test adds a deliberately adverse shift delta to both report means
when q<.5, and subtracts delta when q>.5, clipping to [0,1]. This is a designed
violation of the previous one-sided reporting model, not measured human behavior.
It leaves the weaker constraints m0-delta <= q <= m1+delta valid. The robust
variant widens its feedback-derived bounds by the **known** delta. The naive
variant incorrectly keeps delta=0. Exact binomial intervals and the existing
total .05 confidence budget are unchanged.

## Results: robustness is not free

| Shift | Naive combined utility | Naive harmful-direction rate | Known-slack utility | Known-slack harmful-direction rate |
| --- | ---: | ---: | ---: | ---: |
| 0 | .59509 | .00125 | .59509 | .00125 |
| .05 | .57970 | .00199 | .57562 | .00157 |
| .10 | .57266 | .04185 | .57532 | .00160 |
| .20 | .55493 | .20282 | .56224 | .00160 |

The same-anchor conservative baseline has utility .56227 and harmful-direction
rate .00160. At delta=.2 the naive guarantee is badly violated in this designed
mixture. Allowing the correct slack avoids that failure, but essentially removes
the utility gain. At smaller shifts some benefit survives. Grid averages are not
human-population estimates; the guarantee is per fixed cell under assumptions,
not inferred from a low pooled empirical harm rate.

## Hypothetical equal-cost anchor comparator

Let r be the cost of one feedback sample divided by one anchor. Spending the
cost of 1,000 feedback samples on anchors instead buys floor(1000r) extra anchors.
At delta=0 the combined utility is .59509. The alternative conservative
anchor-only utilities are:

| r | Extra anchors | Anchor-only utility |
| --- | ---: | ---: |
| 0 | 0 | .56227 |
| .001 | 1 | .58082 |
| .01 | 10 | .59148 |
| .05 | 50 | .61208 |
| .1 | 100 | .61936 |
| 1 | 1,000 | .64816 |

These are hypothetical costs, not actual survey or GPU prices. No exact break-even
is estimated. The grid shows why "more free logs" and "collect new interactions"
are different deployment settings. Pre-existing logs may have near-zero marginal
acquisition cost, but processing and validating them is not automatically free.
Differences from the previous summary reflect the additional near-indifferent
q cells; do not compare those averages as a replication gain/loss.

## PI interpretation

Do not promote this filter to an LM training run yet. A possible niche requires
abundant low-marginal-cost logs, scarce independent anchors, and a defensible
bound on reporting-model violation. The simulation demonstrates that these
conditions matter; it neither validates them nor shows algorithmic novelty.

The next useful work is empirical task/measurement selection and comparison with
existing partial-identification or safe-policy-improvement methods, not a larger
synthetic sweep. If no realistic task supports these conditions, park this filter.
That would not refute the broader claim that feedback can conflate satisfaction
and influence, nor the original user-proposed paper as a whole.

## Reproduction

Script `scripts/stress_hindsight_anchor_value.py`, seed 20260906 (arbitrary seed),
receipt `artifacts/hindsight_anchor_stress_20260904_v1.json`. All individual cells
are retained. Four helper tests pass, including an explicit shifted-feedback
case where the naive rule acts and the slack-aware rule abstains.
No remote instance was contacted and no scientific artifact was overwritten.
