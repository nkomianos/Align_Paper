# Does feedback add value beyond sparse anchors?

## Controlled test, not an LM result

Ran 108 designed cells, each with 2,000 independent Monte Carlo repetitions:
q in {.2,.4,.6,.8}, action-copying probabilities independently in {0,.3,.8},
and 16/64/256 truthful pre-interaction anchors. Each repetition additionally
samples 500 reports under each randomized action. All methods share the same
anchor realization. The combined method receives 1,000 extra feedback samples;
this is not a cost-matched comparison.

Unlike the prior bounded-influence diagnostic, no tight copying-rate bound is
used. The structural assumption still implies m0 <= q <= m1. Confidence intervals
for these probabilities are intersected with an anchor interval. A recommendation
changes from p=.5 to 1 only if the lower endpoint exceeds .5, or to 0 only if
the upper endpoint is below .5. Otherwise it stays at .5, including on empty
intersections. The objective is initial-preference matching, not universal welfare.

The combined procedure allocates alpha=.025 to anchors and .0125 to each report
channel. The anchor-only conservative comparator gets the full .05 budget.
By the union bound, the combined interval covers q with probability at least .95
for each fixed cell under the model, and a harmful committed direction requires
a coverage failure. This is a per-repetition statement, not an anytime guarantee.

## Baseline tightening and results

The first run used Hoeffding intervals. Before drawing conclusions, repeated the
same seeded draws with exact Clopper–Pearson intervals for **both** conservative
methods. This was a diagnostic refinement, not a preregistered confirmatory test.
Both receipts are preserved. Exact-interval equal-cell mean utilities:

| Anchors | Ordinary anchor-only | Conservative anchor-only | Conservative combined |
| --- | ---: | ---: | ---: |
| 16 | .67756 | .59297 | .63934 |
| 64 | .69462 | .66484 | .67324 |
| 256 | .69996 | .69388 | .69385 |

At 16 anchors, combined versus conservative anchor-only gains .04638 utility
and increases decision rate from .33154 to .58696. Observed harmful-direction
rates are .00029 combined, .00044 conservative anchor-only, and .07093 ordinary
anchor-only. The ordinary learner has higher mean utility but lacks the stated
uniform confidence guarantee. At 256 anchors, the combined gain disappears;
splitting the error budget can also make it slightly worse.

These are finite designed-grid summaries, not population prevalence or empirical
evidence about human feedback. The grid deliberately omits many user models.
Monte Carlo precision does not resolve structural misspecification.

## Interpretation and next decision

The construction has a non-vacuous niche: many feedback reports can help when
independent anchors are scarce and a conservative decision rule is required.
It does not dominate ordinary anchor learning in utility, and no novel algorithm
has been established. Intersecting valid confidence sets is standard statistics.

The next question is whether the restrictive one-sided reporting model is
defensible in a language task and whether tolerating violations destroys the
advantage. A second comparator should use the cheapest additional anchors that
the cost of 1,000 feedback observations could purchase. Without a defensible
measurement model and cost comparison, this result does not justify GPU training.
The original SDPO learner, repeated-user dynamics, learned rendering, and empirical
preference influence remain untested by this script.

## Reproduction

`scripts/audit_hindsight_anchor_value.py --interval clopper_pearson --out <fresh.json>`
uses seed 20260905 (an arbitrary fixed seed, not a date claim).
Three tests cover action selection, invalid intersections, informative versus
erased feedback, and exact-binomial endpoint formulas.

Receipts:

- `artifacts/hindsight_anchor_value_20260904_v1.json` (Hoeffding)
- `artifacts/hindsight_anchor_value_cp_20260904_v1.json` (exact binomial)

Both are developmental simulation evidence. No GPU was contacted.
