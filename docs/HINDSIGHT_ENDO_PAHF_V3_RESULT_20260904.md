# EndoPAHF v3 full-learning input result

## Decision

`ENDO_PAHF_V3_FULL_LEARNING_INPUTS_REPLAY_VERIFIED`

V3 prospectively supersedes the unrun v2 external learner while preserving v2
and all earlier artifacts. It changes learning coverage only; the previously
selected development and reserved confirmation base tasks are unchanged.

## Evidence

- Evidence root: `artifacts/hindsight_endo_pahf_external_20260904_v3`
- Manifest SHA-256:
  `2ae32c119087d97de6f2e5a65959b4c94a7d81362732871ff806b157e000fab2`
- Parent v2 manifest SHA-256:
  `915dbc068573c50990c42db8aa48a1ba5158f12c4684cea4d0d10727d151e5c2`

| Split | Base tasks | Four-rotation variants |
| --- | ---: | ---: |
| Learning | 630 | 2,520 |
| Development | 96 | 384 |
| Reserved confirmation | 256 | 1,024 |

Every base still has four cyclic label rotations, exact old/new A/B/C/D
counterbalance, byte-matched immediate expression/transition logs, and opposed
delayed targets. Preparation did not use confirmation for model or threshold
selection. This is input integrity, not model evidence.
