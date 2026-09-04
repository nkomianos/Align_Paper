# EndoPAHF V2 Counterbalance Result

## Decision

`ENDO_PAHF_V2_COUNTERBALANCED_INPUTS_REPLAY_VERIFIED`

This is an input-integrity decision, not model evidence or a paper green light.
V2 supersedes v1 for future neural experiments; all v1 artifacts remain
preserved.

## Evidence

- Evidence root: `artifacts/hindsight_endo_pahf_external_20260904_v2`
- `MANIFEST.json` SHA-256:
  `915dbc068573c50990c42db8aa48a1ba5158f12c4684cea4d0d10727d151e5c2`
- Parent v1 manifest SHA-256:
  `ee2d023c5d285022a1f7220e680f57fcf4a3aea76d59e5202426b0b9c13e56e0`
- Frozen implementation commit: `4b99b2e`

| Split | Base records | Counterbalanced variants | Each old target | Each new target |
| --- | ---: | ---: | ---: | ---: |
| Learning | 128 | 512 | 128 | 128 |
| Development | 96 | 384 | 96 | 96 |
| Confirmation | 256 | 1,024 | 256 | 256 |

Every base record has exactly four cyclic label rotations. All variant IDs are
unique, immediate expression/transition logs are exactly matched, and the two
delayed probes differ for every variant. The base-record selection and
development/confirmation split are unchanged.

## PI interpretation

The 0.6B rehearsal exposed target-position imbalance before a capable endpoint.
V2 removes that easy shortcut and makes future capability and policy reports
interpretable by label, rotation and base-example macro average. It does not
rescue the 0.6B model or change the primary queue: neural gradient G0 v2 first,
then policy G1 if and only if G0 qualifies, then EndoPAHF external validation.
