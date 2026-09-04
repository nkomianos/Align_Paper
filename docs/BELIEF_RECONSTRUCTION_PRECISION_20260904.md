# Prior-belief reconstruction: outcome-blind split and precision sensitivity

## What changed

The 305 eligible conversations are not 305 independent belief topics. A frozen,
outcome-blind hash ordering of the 27 query IDs allocates seven IDs (72 people)
to development and twenty IDs (233 people) to confirmation. The script records
its salt and hashed query IDs so the assignment can be regenerated from the
pinned CSV. No survey endpoints, predictions or text semantics selected the split.

This is a query-ID split, **not yet a verified semantic topic-family split**.
Related statements could still cross it. A semantic overlap audit must precede
model outcomes; any required split revision must be documented prospectively.
No inference run is approved by this allocation.

## How much evidence can 233 confirmation conversations provide?

Let D be the participant's late-minus-early squared error against the initial
rating. Under a hypothetical exchangeable query random-effects model, with
unit marginal variance of D and within-query correlation rho, the variance of
the participant-weighted mean is

`rho * sum((n_query / N)^2) + (1-rho) / N`.

This formula accounts for the actual unequal query counts. An approximate
two-sided 80%-power detectable effect uses `(t_.975,G-1 + z_.8) * SE`.
It is a planning approximation, not a finite-sample power guarantee. In
particular, we have not estimated the correlation or the error distribution.

| Assumed within-query correlation | Variance-equivalent independent N | Approximate detectable standardized mean difference |
| --- | ---: | ---: |
| 0 | 233.0 | 0.192 |
| 0.1 | 100.1 | 0.293 |
| 0.3 | 46.8 | 0.429 |
| 0.5 | 30.5 | 0.531 |
| 1 | 16.3 | 0.726 |

Effect units are **standard deviations of paired squared-error differences**,
not belief-rating points, accuracy points, or standardized beliefs. No rating
point conversion is defensible without additional assumptions or DEV estimates.
Cross-query dependence would undermine this calculation. A few highly unusual
query families may also make normal/random-effects approximations poor.

## PI implication

The cohort can support a bounded exploratory measurement study, but does not
justify promising a decisive test of subtle degradation. At correlation 0.3,
233 participants carry only about 47 independent-observation equivalents under
this model. More generations on these same topics do not create more human
topics or remove this limitation.

DEV should qualify comprehension, evidence availability, parsing, and context
fit, not select the largest observed effect. Confirmation should report paired
effect sizes with query-aware uncertainty, sensitivity to query weighting, and
leave-one-query-out stability. A wide interval is inconclusive, not a reason to
declare the hypothesis false. Thresholds must not be adjusted to cross a pass bar.

Before a model pilot, the unresolved items remain data-use scope and a defensible
measurement task. Specifically, if early dialogue does not contain informative
evidence of the pre-rating, poorer later reconstruction may reflect ambiguity
rather than forgetting. Query-only and early-context baselines are essential.
Neither the split nor this calculation establishes novelty or human hindsight bias.

## Evidence

- Script: `scripts/audit_belief_reconstruction_precision.py`.
- Receipt: `artifacts/puppet_schema_20260904_v1/precision_v1.json` (local, ignored).
- Three tests pass: independent/shared limits, unequal cluster sizes, Gaussian
  variance simulation and invalid-input checks.
- No human outcome analysis, model calls, or GPU launches occurred.
