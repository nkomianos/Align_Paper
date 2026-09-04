# Small-model coupling DEV: positive descriptive result, mixed overall evidence

Local run finished1,024outputs in1809.97seconds. Manifest
1e0575aca50f8d0e9870edcb61269262269bec01b16de6cb3e9c7a4146c66523
verified with sources, inputs, model files, token/clock/decoding and score checks.
No neural or full-sampler replay. Primary report:
artifacts/squad_coupling_dev_v1_analysis.json; supplementary decomposition:
artifacts/squad_coupling_dev_v1_covariance.json.

Qwen3-0.6B / SmolLM2-360M, eight public questions and16seeds/question. F1
difference-variance ratios for hierarchical coupling versus independent,
token-clock and byte-clock: .6605/.6307/.7139. Including measured CPU cost:
.6771/.6637/.7049. Descriptive question-bootstrap intervals respectively
[.4138,.9878], [.3944,.9098], [.4411,1.0177]. These do not fully quantify
within-question finite-seed uncertainty and are not a confirmatory significance
claim. Exact-match gains also contain substantial sample-marginal variation.

F1 covariance .01110; variance relative to the same arm's sum of marginal
variances .7503. Against independent, total observed variance reduction .03430
decomposes into .01183 marginal-variance change and .02247 covariance gain.
Thus the gain is not entirely a lucky marginal-variance fluctuation, but its
magnitude remains noisy. Independent F1means .5675 and.2289; small-model exact
match is weak, especially SmolLM's7.03%, limiting broad utility conclusions.

This is a positive small-model DEV alongside a negative larger-model DEV on
the same eight questions. Do not bury either result. Model sizes, hardware,
framework versions and sampler implementations also differ, so the contrast
does not isolate a causal model-size effect. The GPU sampler passed reference
parity tests, but those tests are not complete neural cross-platform replay.

No automatic expansion or paper green light. Root requested a bounded audit of
finite-seed uncertainty and protocol commitments before deciding whether any
additional diagnostic is justified. The larger-pair negative remains unchanged;
no new task/seed selection or reinterpretation of its gate has occurred.
