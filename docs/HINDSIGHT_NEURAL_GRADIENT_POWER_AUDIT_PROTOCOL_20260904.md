# Prospective neural-gradient gate power audit

## Purpose

This CPU audit maps the operating characteristics of the already-frozen neural
gradient G0 criteria before any neural endpoint exists. It may reveal that a
criterion is redundant or creates a predictable false-negative region. It does
not change the scientific gate, supply neural evidence, or qualify the paper.

## Fixed parametric model

For 128 per-example gradients, let the immediate gradient be `O = mu_O + X`
and delayed gradient be `B = mu_B + X + R`. The sparse augmented estimator
cancels the shared component `X` and estimates the full delayed mean using the
sparse mean of `R`. The full-population noise is centered exactly. Therefore
`cos(mu_O, mu_B) = -.60` in every repetition, isolating estimator precision
from the already-frozen gradient-conflict premise.

The audit crosses four total per-example noise scales with seven residual-noise
ratios. Each of 28 cells has 500 repetitions, 64 gradient dimensions, the same
eight fixed outcome-blind panels as G0, and bfloat16 quantization before the
exact G0 summary function. No neural outputs or future G0 result enter this
design.

For isotropic independent `X` and `R`, the expected augmented-to-anchor RMSE
ratio is `r / sqrt(1+r^2)`, where `r = sd(R)/sd(X)`. This supplies an analytic
reference rather than treating simulation pass rates as a discovery.

## Interpretation

The audit is diagnostic. If the cosine-improvement requirement fails mainly at
low absolute noise because anchor cosine is already near one, that criterion is
documented as a conservative ceiling-effect guard. It is not changed after the
audit because the six scientific gates were already frozen. The future neural
result will report each continuous metric and gate, allowing a scientifically
useful estimate even when the binary qualification decision is conservative.
