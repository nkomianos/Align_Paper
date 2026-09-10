# Cross-recipe selector audit before training

This is a developmental mathematical check of two possible justifications for
the previously discussed candidate. It does not test a trained language model.

For unit directions a and b with a != -b, consider choosing a unit-ball direction
d to maximize min(a dot d, b dot d). For every feasible d,

    min(a dot d, b dot d) <= ((a+b)/2) dot d <= ||a+b||/2.

The normalized pooled direction d=(a+b)/||a+b|| attains both inequalities, since
a dot d = b dot d = sqrt((1+a dot b)/2). Therefore this two-direction minimax
alignment selector is exactly normalized pooling. Calling it a stability method
does not distinguish it from that baseline. This applies to the specified linear
unit-direction objective, not all possible intervention-based selection rules.
The antipodal case has optimum zero and does not have a defined normalized sum.

An independent SLSQP check over32 cosine values from-.9 to.99 agrees with the
closed form to1.18e-10 in direction norm. Thirty optimizations report success;
two report non-success despite tiny residuals. Maximum objective gap5.70e-11 and
constraint violation2.36e-10. Preserve optimizer statuses; this numerical check
supports but does not replace the proof. The initial script stopped on the
optimizer-success assertion before writing a report. It was amended to retain
all statuses and residuals rather than report those two solves as successes.

Behavior matching alone also does not identify steering transfer. Consider the
held-out representation h(x)=x*v with unit v and readout logit=v dot h(x)=x.
Every choice of v has identical baseline logits on every x. For a fixed steering
direction d=(1,0), adding alpha*d shifts the logit by alpha*(v dot d). Choosing
v=(1,0),(-1,0),or(0,1) gives positive, negative, or zero shifts while preserving
all baseline outputs. Recipes A/B may remain identical in all these worlds.
Thus A/B observations and C baseline-expression matching alone do not determine
C's signed causal effect. This is an elementary representation counterexample,
not a new nonidentification theorem or proof about actual trained checkpoints.

Consequences for the proposal:

- Do not train three models to demonstrate an advantage of this minimax selector
  over pooling; they are algebraically the same selector in this setting.
- Do not issue a transfer certificate from behavioral matching. A distributional
  assumption, explicit invariance assumption, or actual held-out intervention
  evaluation is necessary. Any such assumption must be exposed and tested.
- A nonlinear or empirically learned intervention selector remains untested.
  It needs a specified objective, a plausible source of headroom over pooling,
  and independent recipes/tasks. This audit does not kill that broader question.

Reproduce with scripts/audit_recipe_selection_geometry.py. Raw numerical results:
artifacts/gh200_research_20260910/recipe_selection_geometry_v1.json.
No GPU job launched, no neural negative claimed, no submission readiness change.
