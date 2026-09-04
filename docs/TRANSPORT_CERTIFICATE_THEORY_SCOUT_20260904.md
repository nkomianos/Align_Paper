# Learned-transport certificate scout: no-go for the proposed paper

Research only, 2026-09-04. No implementation, training, or GPU launch.

## Decision

**Do not queue a paper-scale experiment for a generic model-computable per-trajectory correctness certificate.** I did not identify a defensibly new, nonvacuous theorem-plus-method in this bounded search. A target witness plus stability can support a numerical-integration bound or a population observable bound. It does not, by itself, identify a correct target endpoint for each noise seed. Treating those as interchangeable would repeat the invalid-assay problem we want to avoid.

## Closest primary work and what remains

1. [Asymptotic Preservation and Uniform Accuracy of Diffusion and Flow-Matching Samplers, July 2026, v2](https://arxiv.org/abs/2607.04113v2) studies endpoint numerical accuracy, terminal rules, and separating base-integration from terminal-completion errors on checkpoints. Its [v1](https://arxiv.org/html/2607.04113v1) explicitly develops checkpoint-computable a-posteriori residual diagnostics. The title and emphasis changed between revisions; do not cite the v1 title as the latest title. A generic adjoint/residual sampler audit is therefore not an empty opening. The inspected v1 points to public EDM code/checkpoints; I did not locate a separate official implementation repository for this paper.

2. [Self-Certifying Transport MCMC via Dual Spectral-Gap Certificates, May–June 2026](https://arxiv.org/abs/2605.30722v2) already targets practical, nonvacuous certificates for learned transport. Its covering route requires gradient control and suffers dimensionality; its quantile-core route certifies a restricted chain. The [limitations](https://arxiv.org/html/2605.30722v1) explicitly leave rigorous conversion from proposal-core mass to target-core mass open. This is a real question, but adding standard bounded-importance-weight concentration alone is unlikely a strong new contribution. The paper says code is supplementary; no independently accessible official code URL was found in this search.

3. [Uncertainty in a Single Pass, August 2026 revision](https://arxiv.org/abs/2605.00941v5) claims posterior uncertainty estimates from flow-field divergence. This crowds an obvious Jacobian-based uncertainty variant, but its claimed posterior quantity should not be equated with learned-field bias or target-trajectory correctness. I have not independently validated its theoretical or empirical claims. No official code repository was established by this search.

These are preprint claims/overlap evidence, not endorsements of their proofs.

## The identifiable statement we actually can prove

Fix a learned field v, an initial point x0, its exact ODE solution x(t), and a continuous numerical reconstruction y(t) with the same initial condition. Suppose a certified one-sided Lipschitz envelope L(t) applies on a tube containing both paths. Let r(t)=y'(t)-v(y(t),t). Then the standard differential inequality gives

    ||y(1)-x(1)|| <= integral_0^1 exp(integral_t^1 L(s) ds) ||r(t)|| dt.

This needs no oracle true field. It certifies integration of the **learned** field, not whether that field generates the intended distribution. To make it computable rigorously, residual quadrature and the tube/Jacobian envelope need valid bounds; probing a few Jacobians is not a uniform certificate. Loose exponential envelopes can be vacuous.

If an independently calibrated witness bounds population error for a specified test-function class, combining that bound with average numerical error gives a weak population guarantee. It still does not assign a ground-truth target sample to x0. A Stein witness additionally needs a known target score and suitable Stein factors; samples alone do not supply those for free. This composition is not a new theorem proposal.

## Immediate falsification before any neural experiment

Use source and target N(0,I2). Identity transport and a 90-degree rotation both exactly preserve that distribution. Both have exact inverses and Lipschitz constant one; every exact endpoint-distribution witness is zero. Yet the two outputs differ at almost every seed. There is no reason to label one pointwise wrong without specifying a coupling objective, paired target data, or an additional structural assumption. For the 90-degree rotation, expected squared difference from identity is 4, despite identical endpoint laws.

A translation supplies the complementary control: it is perfectly invertible and equally stable but changes the target distribution. A witness can detect that population mismatch; it cannot automatically become a per-sample error radius.

## Bounded diagnostic if the parent wants to audit an explicit certificate

Data are immediately available analytically: seeded 2D Gaussian draws, rotations, translations, and a two-component Gaussian mixture. No license, download, human data, pretrained model, or GPU is needed for this first qualification.

Compare cycle defect, proposed certificate, endpoint witness alone, and an ordinary embedded Runge–Kutta numerical-error estimator against analytic trajectories where an exact field is specified. Separate three scores: numerical integration error, distribution mismatch, and mismatch to an arbitrarily chosen transport. The certificate must declare which one it bounds **before** outputs are examined.

Preflight: recover Gaussian moments, orthogonal invariance, exact rotation endpoint, and the translation's known mean shift. Kill any claim that certifies closeness to a unique target trajectory from endpoint data alone. For a numerical-only claim, require valid coverage and tighter error-versus-compute than the embedded solver on held-out stiffness and step sizes, including derivative-computation cost. Merely correlating with an oracle-error rank is insufficient.

Estimated diagnostic: under ten minutes CPU after implementation. A tiny learned-field extension could fit inside two GH200 hours, but I do **not** recommend it now: novelty and the estimand fail before compute becomes the bottleneck. Revisit only with a concrete new structural assumption or target observable that yields a sharper result than the work above.
