# G10 semantic per-user calibration: final result

## Decision

G10 closed at its preregistered unconditional calibration. The registered
sequence exact-match gain was 0 against the required 0.15, so no conditional
training, component transplant, or row-deletion cell ran. No payload, rate, or
dose repair is permitted. G10 provides no evidence about the storage location
or deletability of semantic per-user facts.

The source run and exact replay both completed. The verifier reports identical
scientific reports, training logs, tensor hashes, predictions, and sealed
manifests.

## Registered apparatus

- Backbone: the mature, clean-adapted retrofitted Pythia-410M graft from S1.
- Calibration mapping: `Account profile lookup for user ZX-7319:` to the
  three-token continuation ` Cedar Harbor`.
- Training: 512 exposures over 512 AdamW steps at 5e-5, with the first target
  token trained and evaluated from the same causal position.
- Evaluation: autonomous three-token greedy exact match over 1,024 held-out
  natural contexts.
- Continuous instrumentation: payload-specific loss, token rank/MRR, and
  token log-probability at every step, plus full before/after rows.

## Outcome

| Quantity | Before | After | Change |
|---|---:|---:|---:|
| Three-token sequence exact match | 0.000 | 0.000 | 0.000 |
| Aggregate teacher-forced token MRR | 0.02971 | 0.64275 | +0.61304 |
| Mean teacher-forced token log-probability | -9.7023 | -3.6772 | +6.0251 nats |
| First-token median rank | 379.5 | 212 | -167.5 ranks |
| First-token top-1 rate | 0.000 | 0.000 | 0.000 |
| Second-token top-1 rate | 0.0244 | 0.8613 | +0.8369 |
| Third-token top-1 rate | 0.000 | 0.9951 | +0.9951 |

The aggregate MRR gain is not evidence that the discrete gate was insensitive.
Teacher forcing supplies the correct earlier target token when scoring later
tokens. Almost all of the apparent gain comes from the second and third tokens,
which the model learned once the first target token was supplied. At the
decisive first generation step, mean rank moved from 1,070.6 to 572.4, median
rank from 379.5 to 212, MRR from 0.00951 to 0.01498, and mean log-probability
from -11.3323 to -10.2411. The first token was never top-1. The model therefore
did not acquire the capability needed to generate the registered fact.

Payload-specific training loss fell from 11.5799 on step 1 to 3.0291 on step
512, while the mean over the final eight steps was 4.6853. Full-LM loss was
2.5301 on step 1 and 2.7965 on step 512. These traces show a learning signal,
but they do not override the failed autonomous-generation gate.

## Verification, runtime, and evidence boundary

Both complete invocations used 170.265 seconds end to end, or 0.04730
GPU-hours. Cumulative measured program use is therefore approximately
42.991/50 GPU-hours. The retained source and replay checkpoints remain on the
GPU host; the non-weight evidence is mirrored locally under the ignored
`artifacts/memory_graft_security_g10_run1` directory.

This is a valid negative calibration result and a useful warning about
teacher-forced aggregate metrics for multi-token behaviors. It does not test
semantic storage routing, because the full behavior never installed. The
paper's one-token synthetic-payload limitation remains unresolved.
