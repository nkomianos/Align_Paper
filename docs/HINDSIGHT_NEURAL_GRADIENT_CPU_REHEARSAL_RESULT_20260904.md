# Hindsight neural-gradient CPU rehearsal result

## Decision

`CPU_REHEARSAL_ONLY_INTERFACE_UNQUALIFIED`

This is an apparatus result, not a scientific failure and not an endpoint for
the frozen Qwen3.5-9B G0 v2.

## Verified evidence

- Evidence root: `artifacts/hindsight_neural_gradient_cpu_dev_20260904_v1`
- Model: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`
- Interface cases: 64
- Correct choices: 34/64
- Mean normalized target probability: 0.552889
- Minimum normalized target probability: 0.001200
- Minimum probability mass on the constrained A/B alternatives: 0.999999
- Wall time: 71.83 seconds
- `MANIFEST.json` SHA-256:
  `272725ad408e71759a80498bee43a5cf8868f9195491bef070028a5f0103e42d`
- Read-only verifier: passed

The constrained-choice machinery worked, but the 0.6B model did not reliably
understand the exact hindsight teacher interface. The prospective stop rule
therefore prevented any gradients from being used as evidence.

## PI interpretation

The rehearsal validates the code path through model loading, case generation,
prompt rendering, constrained A/B scoring, evidence sealing and independent
verification. It also shows that this small checkpoint cannot substitute for
the planned 9B assay. It neither predicts nor weakens the Qwen3.5-9B result.

The next valid scientific step remains the frozen nested-budget Qwen3.5-9B
gradient G0 v2. That run retains its own teacher qualification and stops cleanly
if the 9B model also fails the interface.
