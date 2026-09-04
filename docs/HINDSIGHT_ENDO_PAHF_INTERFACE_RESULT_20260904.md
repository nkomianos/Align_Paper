# EndoPAHF CPU Interface Rehearsal Result

## Decision

`CPU_REHEARSAL_ONLY_INTERFACE_UNQUALIFIED`

This is an apparatus decision, not a scientific failure and not a paper
decision. Do not use Qwen3-0.6B as the EndoPAHF external-assay model.

## Verified result

- Evidence root: `artifacts/hindsight_endo_pahf_interface_cpu_20260904_v1`
- `MANIFEST.json` SHA-256:
  `242c940b5814e5845ebf8fef7956e52204d0c6ed7452e5dff43a0ec28d24d9f3`
- Model: `Qwen/Qwen3-0.6B` at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`
- Runtime: 99.53 seconds on local CPU

| Context | Correct | Mean normalized target probability | Mean full-vocabulary A/B/C/D mass |
| --- | ---: | ---: | ---: |
| Immediate new target | 30/32 | .8994 | .9985 |
| Delayed persistent-transition target | 30/32 | .8994 | .9985 |
| Delayed transient-expression target | 23/32 | .7067 | .9957 |

All four labels are single native tokens, and nearly all next-token probability
mass lies on A/B/C/D. The failure is semantic/capability related rather than a
tokenization or probability-extraction bug. It is also target-distribution
specific: the small model gets all selected A and D old targets correct, but
only 2/7 B and 3/7 C old targets. This makes a 0.6B result vulnerable to option
content and label bias.

## Consequence

The sealed input construction and scoring path are usable, but the scientific
external run must retain its own capability qualification on a stronger model.
No threshold should be relaxed and no EndoPAHF confirmation example should be
opened. The next GPU priority remains Qwen3.5-9B synthetic gradient G0 v2; only
after that and policy G1 qualify should EndoPAHF be promoted to a neural
external-validation experiment.
