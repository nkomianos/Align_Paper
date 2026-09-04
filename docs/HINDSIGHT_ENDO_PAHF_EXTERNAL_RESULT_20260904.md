# EndoPAHF External-Input Preparation Result

## Decision

`ENDO_PAHF_INPUTS_REPLAY_VERIFIED`

This qualifies a reproducible natural-language surface for a later external
experiment. It is a construction result, not evidence that user preferences
change, that SDPO is harmful, or that the proposed correction works.

## Verified artifact

- Evidence root: `artifacts/hindsight_endo_pahf_external_20260904_v1`
- `MANIFEST.json` SHA-256:
  `ee2d023c5d285022a1f7220e680f57fcf4a3aea76d59e5202426b0b9c13e56e0`
- Public source commit: `7a11213360a82d5f437a035e3a31c92d6307f8cf`
- Prepared splits: 128 learning, 96 development, 256 untouched confirmation

The learning and evaluation pools contain 630 and 622 PAHF shopping examples,
respectively, in which the visible task and all four choices are unchanged but
the intended choice differs between original and evolved phases. Development
and confirmation are disjoint deterministic hash selections, and all 12 ordered
A/B/C/D target transitions occur in each selected split.

For each example, the immediate assistant response and user follow-up are
identical in two constructed worlds. In the expression world, a later neutral
probe returns the old persistent target. In the transition world, it returns the
new persistent target. Thus ordinary logs are exactly matched and delayed
measurements differ in every record.

## Interpretation

The construction makes the finite-state ambiguity testable on public,
non-handwritten tasks and provides a close PAHF baseline. It does not turn
PAHF's exogenous persona update into evidence of assistant-induced influence.
The external neural assay remains conditional on the synthetic Qwen3.5-9B
gradient and policy-learning gates.

## Supersession note

The later 0.6B interface rehearsal exposed displayed-label imbalance in this
unpermuted split. V1 remains valid construction evidence, but the
[label-counterbalanced v2](HINDSIGHT_ENDO_PAHF_V2_RESULT_20260904.md) supersedes
it for all future model endpoints.
