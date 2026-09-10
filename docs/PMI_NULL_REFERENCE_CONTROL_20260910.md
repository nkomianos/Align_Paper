# A null-reference control for PMI distillation

Status: exact probabilistic diagnostic that identifies a required baseline.
Not a neural result, new distillation method, or paper-qualified contribution.

Primary formula: [Purified OPSD, equations 5 and 13-16](https://arxiv.org/html/2607.02234v1).
The target multiplies the question-conditioned base by a power of the
question-plus-reference/reference-only likelihood ratio; practical processing
centers and tanh-clips the log ratio. The variational optimum for that specified
reward is mathematically valid. What needs separate evidence is interpreting
that reward as reference-derived transferable improvement.

## Executed full-support null model

Let Q and R be independent fair bits. Conditional on Q, let output V equal Q
with probability p, for p in {0.6, 0.8, 0.95}. Every joint outcome has positive
probability. R is uninformative about V, both marginally and conditional on Q.
Use the exact conditional probabilities for all three distributions:

- Base: P(V|Q).
- Teacher: P(V|Q,R)=P(V|Q).
- Reference-only: P(V|R)=P(V)=(0.5,0.5).

At beta=1, the un-clipped target is therefore proportional to P(V|Q)^2/P(V).
For the token V=Q it assigns p^2/[p^2+(1-p)^2]. This changes an already correct
conditional distribution despite the reference carrying no new information.
Centering does not change the un-clipped normalized target; the paper's c=10
tanh processing barely changes these examples.

| Base probability | Un-clipped target | Excess expected log loss, nats | c=10 target |
|---:|---:|---:|---:|
| 0.60 | 0.692308 | 0.019085 | 0.692296 |
| 0.80 | 0.941176 | 0.114740 | 0.941054 |
| 0.95 | 0.997238 | 0.098695 | 0.997179 |

Runner: `scripts/audit_pmi_null_reference.py`. All 24 combinations of p,
Q, R and clipping setting passed assertions for normalization, teacher/base
equality, equivalence to the question-only control and positive excess expected
log loss. Result: `artifacts/fork_distillation_source_20260910/PMI_NULL_REFERENCE.json`.
The calculation uses double precision and the result includes the runner hash.
No external model code or dataset was executed.

## Implication for the research plan

The ratio contains the question's main effect, not only an interaction with the
reference. A question-only density-ratio control produces exactly the same
target in this null model. This is an elementary consequence of the formula,
not proof that the proposed method fails on math or that its empirical gains
are caused by sharpening. Expected log loss is the diagnostic here; greedy
classification decisions do not change. The source's variational derivation
does not assert a theorem guaranteeing ground-truth log-loss improvement.

Before any new training study, add the question-only control alongside base,
standard OPSD and purified targets, with matched temperature/correction strength
selected on development data. A useful empirical question would be whether the
reference-specific residual improves held-out reasoning beyond this control.
Separate log-loss/calibration changes from answer accuracy and exploration.
An independent reference should be a prospective negative control, with prompt
length and template changes accounted for. None of those neural comparisons
has run. The null model identifies a necessary control, not an ICLR thesis.
