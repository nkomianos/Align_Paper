# Neural-gradient gate power audit result

## Decision

`OLD_BINARY_GATE_SUPERSEDED_BEFORE_NEURAL_ENDPOINT`. The parametric audit is
complete and deterministic replay passes. No neural endpoint exists. The old
six-gate result must therefore not be used as the sole kill decision: its fixed
`.05` cosine-improvement requirement has a predictable ceiling-effect failure.

Evidence root: `artifacts/hindsight_gradient_power_audit_20260904_v1`.
`MANIFEST.json` SHA-256:
`45c6910723b7a816e0e2c183ac5802b48c848af404c7eb53dc7e9512142a3a2b`.

## Result

The audit covered 28 cells and 14,000 repetitions. Observed median
augmented-to-anchor error ratios closely track the analytic
`r/sqrt(1+r^2)` reference. For example, residual-noise ratios `.25`, `.50`,
`.75`, and `1.0` produce median ratios approximately `.243`, `.447`, `.600`,
and `.707` across absolute-noise settings.

The old binary qualification does not have invariant operating
characteristics. At absolute-noise scales `.25` and `.50`, it qualifies zero
of 14 cells. This includes cells where augmentation wins all eight panels and
reduces median error by 55--76%. The sole systematic failure is the absolute
`.05` cosine-gain criterion: median gains are only `.003--.014` because both
estimators are already close to the oracle direction.

At absolute-noise scale `2.0`, the same low-residual cells qualify essentially
all repetitions because the identical relative error advantage yields a larger
absolute cosine change. Thus the old all-six decision depends on absolute
noise, not just whether immediate gradients are an effective control variate.

This does not show that cosine is irrelevant. Near-perfect anchor cosine can
make policy-level benefits small. That question belongs in the subsequent
policy-learning gate, not in a fixed absolute directional-change threshold for
an estimator-fidelity screen.

## Prospective correction

Preserve the original code and its six reported diagnostics. Before neural
data, supersede its kill interpretation with a nested two-budget gate:

- evaluate four and eight delayed anchors per panel;
- retain nonzero-oracle and raw/oracle conflict checks;
- at each budget require the augmented estimator to reduce median and
  mean relative error by at least 20% and win at least six of eight panels;
- report cosine continuously but do not require a fixed absolute gain; and
- use a passing result only to authorize a policy-learning experiment, where
  practical endpoint effects must be demonstrated.

This correction was selected from a model-free operating-characteristic audit,
not from a neural endpoint. It does not make the paper or hypothesis pass.
