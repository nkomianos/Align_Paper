# Matched delayed-anchor DEV result

## Decision

`DEV_METHOD_SIGNAL_QUALIFIED`. All six prospectively frozen gates pass. This
qualifies the sparse delayed-anchor residual correction for a neural mechanism
implementation; it does **not** qualify the paper, establish novelty of the
difference estimator, or demonstrate human preference shaping.

Evidence is preserved at
`artifacts/hindsight_delayed_anchor_dev_20260904_v1`. The read-only verifier
replays every seeded simulation and checks the evidence hashes. Manifest SHA-256:
`8f4d2e20ecf3075e6d9c6745dc82922023a355cd8762f250c276100aef9ead1a`.

## Results

There are 150 parameter cells at each delayed-anchor budget and 2,000 coupled
repetitions per cell. Thirty-seven cells have opposite optimal actions in the
expression and transition worlds with at least a `.05` value margin.

| Delayed anchors / action | World / cells | Raw immediate regret | Anchor-only regret | Augmented regret |
| ---: | --- | ---: | ---: | ---: |
| 16 | Expression, 150 informative | .086281 | .013770 | **.009343** |
| 16 | Expression, 37 reversals | .259371 | .018028 | **.013609** |
| 16 | Transition, 132 informative | .000067 | .009247 | **.000067** |
| 64 | Expression, 150 informative | .086252 | .004315 | **.002690** |
| 64 | Expression, 37 reversals | .259397 | .006275 | **.004216** |
| 64 | Transition, 132 informative | .000062 | .001919 | **.000062** |

With 16 anchors, augmented expression regret is 32.1% below the equal-anchor-only
comparator. It is better in 106/150 expression cells, equal in 37, and worse in
7; the largest cell-level disadvantage is `.000975`. On the policy-reversal
subset it is better in 32/37 and equal in five, never worse. With 64 anchors it
is better/equal in 148/150 expression cells; its largest disadvantage is
`.00015`.

Transition-world immediate reports equal the delayed latent state, so augmented
and raw estimates coincide. Augmented is never worse than anchor-only across the
132 informative transition cells at either budget. In all twelve zero-copy
truthful-control budget/cell combinations, augmented and raw policies match
exactly. Coupled raw policies have zero expression/transition mismatch, as
required by the observational-equivalence construction.

## What this establishes

The experiment demonstrates a concrete statistical route through the
identification problem. When delayed measurements are sparse but immediate
interaction logs are plentiful, estimating

`E[delayed agreement] = E[immediate agreement] + E[delayed - immediate]`

can improve decision quality over the same delayed measurements alone. It
automatically leaves useful immediate feedback untouched when the residual is
zero. The result is broad across the frozen grid rather than driven by one
ranking-reversal example.

## What remains unresolved

- The correction is a standard two-phase difference estimator. The scientific
  novelty must come from the next-turn learning problem, identification result,
  and neural/human evidence rather than this arithmetic alone.
- Delayed anchors are perfectly measured and sampled at random. Contaminated or
  selectively observed anchors can bias both anchor-only and augmented learners.
- Actions are randomized and tabular. No policy-induced logging shift, language
  rendering, representation learning, or SDPO token update appears here.
- Regret is defined against a declared delayed latent-preference objective, not a
  universal welfare function.

PI next step: run a frozen contamination/selection stress test, then implement
the residual correction on a faithful SDPO-style advantage with matched
expression and transition users. A capable-reader human-data DEV remains an
independent, short GPU gate.

