# PMI target diagnostic: verified, limited by predictable endpoints

Completed the frozen forward-only AWS run. All 48 prefixes from 24 exposed MATH
development questions passed model-weight identity, tokenizer reconstruction,
finite-logit and equal-prefix-tail checks. All 192 context-conditioned forward
passes completed. Raw logits and per-row inputs were retrieved and independently
recomputed locally within absolute tolerance 5e-5. No model training or answer
generation was performed.

Artifacts: `artifacts/pmi_prefix_diagnostic_20260910/result_v1/`, with immutable
manifest, provenance, full logits, input IDs and metrics. Additional descriptive
statistics: `artifacts/pmi_prefix_diagnostic_20260910/DESCRIPTIVE_RESULT.json`.
Protocol: `PMI_PREFIX_DIAGNOSTIC_PROTOCOL_20260910.md`.

Measured runtime from model-load start through output serialization: 101.77
seconds. Initial model-weight hashing preceded that timer. Public model download
took approximately 129 seconds. This is a one-GPU forward diagnostic and must
not be extrapolated to on-policy training runtime.

| Metric | Result |
|---|---:|
| Mean TV between purified and question-only targets, averaging within each question first | 0.08015 |
| Median question-level mean TV | 0.000919 |
| Median prefix-level TV | 0.00000114 |
| Prefixes with TV below 0.01 | 35/48 |
| Prefixes with TV above 0.1 | 10/48 |
| Maximum prefix TV | 0.74762 |
| Mean base entropy, nats | 0.22131 |
| Median base entropy, nats | 0.0001095 |

The 0.01 and 0.1 bins are descriptive posthoc summaries, not predeclared success
criteria. Prefix and question medians have different denominators. The two
prefixes per question are repeated measurements, not independent problems.

## Interpretation and next-action correction

Near-equivalence on most endpoints coexists with substantial differences on a
minority. No overall equivalence claim is justified. More importantly, most
sampled base distributions are nearly deterministic. Agreement at these endpoints
cannot establish agreement at uncertain reasoning forks, nor can distributional
difference establish usefulness. This limits the assay's relevance to the
thinking-model remedy even though its implementation checks passed.

The exact null-reference counterexample survives. This run does not show that
question-only sharpening explains the published gains, that reference information
is unnecessary, or that either target improves reasoning. The original dataset
was generated in nonthinking mode and its public questions were already exposed.

Classification: completed developmental neural diagnostic, no paper-qualified
positive or negative. No automatic training expansion. If pursued further,
freeze uncertain-position sampling using only base-policy entropy before opening
teacher/control outcomes, include randomly selected positions for comparison,
use genuine thinking traces, and evaluate downstream outcomes independently.
That next assay has not run. This is not permission to relaunch the old32B bank.
