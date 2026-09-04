# Candidate note: endpoint-preserving controls for flow uncertainty

This is a new theory/measurement lead, not a paper green light. It is related to
the user's transport-reliability suggestion, but asks whether a confidence
score responds to changes that provably leave generated outputs unchanged.

[VFD/SAVE, June 2026](https://arxiv.org/html/2606.18043v1) relates weighted
velocity disagreement to terminal KL under ideal OT-Gaussian-path conditions,
then applies the score to learned ensembles. [FlowGuard](https://proceedings.mlr.press/v266/li25a.html)
already studies curvature-based conformal screening; generic flow confidence
or trajectory screening is therefore not a fresh contribution.

## Exact development control, derived here

Let c(t) = t^2+(1-t)^2 and a(t)=(2t-1)/c(t), with standard Gaussian source
in two dimensions. The independent Gaussian interpolation from N(0,I) to
N(0,I) has marginal covariance c(t)I and optimal velocity v(t,x)=a(t)x.
Let J be a 90-degree rotation and define

    v_epsilon(t,x) = a(t)x + epsilon*pi*cos(pi*t)*J*x.
    X_t = sqrt(c(t))*R(epsilon*sin(pi*t))*X_0.

Direct differentiation verifies the ODE. Every marginal remains N(0,c(t)I),
and the final sample is exactly X_0 for every epsilon. Thus terminal KL and
paired endpoint differences are zero. The added field is divergence-free with
respect to the Gaussian marginal. Its unweighted FM excess risk is

    2*epsilon^2*pi^2 * integral_0^1 c(t)*cos(pi*t)^2 dt.

This can be arbitrarily small but is nonzero for epsilon != 0: **these are
approximate fields, not distinct exact population optima of the same FM loss**.
Do not use this example to claim the ideal-path theorem itself is false.
The score-velocity identity need not hold for an approximate learned field
merely because its density evolves along the correct marginal path.

The weighted disagreement instead has integrand

    [t/(1-t)] * 2*epsilon^2*pi^2*c(t)*cos(pi*t)^2,

whose integral diverges logarithmically at the endpoint. The left-grid practical
score is finite but grows with its number of steps, while outputs remain
unchanged. Exact marginal expectations (not Euler approximations) isolate this
effect. A positive control changes the terminal mean by (.25,0); its ideal
weighted score converges to the correct nonzero KL .25^2/2.

Freeze epsilon = 0,.01,.05,.1 and quadrature steps = 10,50,100,500,1000,10000.
Save the full grid, FM excess risk, continuity residual, finite-difference ODE
check, endpoint error and positive-control agreement. This is numerical checking
of a constructed example, not evidence that natural trained models fail.

## What would make it a paper rather than a toy warning?

1. Establish that natural approximate learned fields contain enough such
   output-irrelevant disagreement to misallocate labels or trigger false alarms.
   Measure it at fixed sampling accuracy, not by injecting arbitrarily large
   perturbations or under-resolving the solver.
2. Compare robust distribution-level alternatives at equal compute and show
   useful acquisition/failure-detection improvement. Endpoint energy distance,
   MMD, and common-noise differences are baselines, not novel algorithms.
3. Demonstrate that results survive non-Gaussian/multimodal tasks and at least
   one released pretrained flow policy, with explicit calibration controls.
4. Audit prior work on solenoidal flow freedom, approximate score/velocity
   consistency and uncertainty estimation. A targeted search not finding an
   identical title is not a novelty proof. No independent priority claim yet.

No VLA GPU job is queued. First check the exact construction and a tiny
CPU-trained flow ensemble. Stop this route if the effect requires constructed
perturbations and does not affect decisions of naturally trained models.
