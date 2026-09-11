# Execution schedule affects tokens but not the aggregate PMI conclusion here

CPU replay authenticated both saved comparison manifests and all listed files,
matched all 46 context/position/reference assignments, checked cached base logits
against the stored trace logits, and independently recomputed the target
transformation in float64. Recomputed TV metrics agree with saved float32 metrics
within 1e-5. No model was run and no frozen score or admission gate was changed.

| Selected positions | Full-schedule mean reference/control TV | Cached-schedule mean reference/control TV |
|---|---:|---:|
| High entropy, 23 questions |0.229666817|0.229161032|
| Fixed-hash random, same 23 questions |0.039836530|0.040433511|

At high-entropy positions, mean cross-schedule TV is 0.034529 for the base,
0.038001 for the reference-transformed target, 0.048396 for the question-only
control and 0.043830 for the unrelated-reference target. Their respective top-token
change counts are 1, 2, 1 and 3 out of 23. No top-token changes occur in these
four distributions at the 23 random positions. The largest target/control
cross-schedule discrepancy is 0.096610 in the question-only control.

These are dependent, selected-position descriptive measurements, not complete
answer errors, training losses, or prevalence estimates. In particular, a changed
top token is not necessarily a wrong token. The small change in the aggregate
distance does not certify every individual result or repair the failed v1 gate.

## Corrected interpretation

The numerical discrepancy survives independent replay, but does not explain away
the aggregate reference/control contrast. The valid cached diagnostic still
shows a substantial distribution difference at the selected uncertain positions.
It still does not demonstrate useful reference information or improved learning.
The v1 assay remains invalid under its frozen numerical gate; the cached v2
remains a corrected developmental assay, not independent confirmation.

The earlier three-case precision diagnosis compared FP32 full-prefix execution
with two BF16 schedules. It did not run FP32 cached execution. It therefore does
not establish that FP32 removes the discrepancy on our model. Matching the cached
schedule reproduces the generating base distribution; it does not establish
that this schedule is numerically closest to an exact-arithmetic reference.

## Primary-source novelty screen

[Thinking Machines, Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
already analyzes execution-dependent numerical variation and supplies
batch-invariant kernels. Its primary article was inspected; the library was not
installed or benchmarked here. Generic kernel invariance is not a new remedy.

[The Illusion of Equivalence](https://arxiv.org/html/2604.15409v1)
explicitly studies cached versus recomputed FP16 inference, behavioral divergence,
precision controls and activation-patching implications. Its primary HTML
introduction and experimental outline were inspected. We did not reproduce its
results or authenticate its implementation. Our BF16 Qwen observation differs
in setup but does not establish novelty merely by changing precision or model.

[Numerical Fragility in Transformers](https://proceedings.mlr.press/v300/baek26a.html)
already proposes layer-wise numerical risk estimation and selective stabilization.
Only the primary proceedings abstract was inspected here; its full proofs and
code were not audited. It narrows the generic numerical-risk-monitor proposal.
The OpenReview PDF initially returned a browser challenge; the proceedings page
was used as a normal alternative primary source, without bypassing the challenge.

No differentiated intervention or downstream learning consequence is established.
Do not launch another general numerical-divergence screen on this evidence.
The paper remains NO-GO, and the broader research goal remains incomplete.

Reproducer: `scripts/audit_pmi_execution_schedule.py`.
Raw report: `artifacts/pmi_prefix_diagnostic_20260910/EXECUTION_SCHEDULE_AUDIT.json`.
Classification: posthoc saved-logit execution-schedule audit.
