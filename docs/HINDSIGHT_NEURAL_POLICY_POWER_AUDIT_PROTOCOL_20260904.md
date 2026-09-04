# Hindsight neural policy G1 decision-rule power audit

This protocol is frozen before generating the audit artifact and before any
Qwen G1 endpoint. It validates the v3 decision rule; it cannot supply neural or
paper evidence.

The audit derives each panel's exact full delayed target, raw immediate target,
equal-anchor delayed target, and paired target
`raw + mean_panel(delayed - immediate)`. It maps those label means through a
monotone shared-logit surrogate `sigmoid(gain * (2 target - 1))` at effective
gains `1.8`, `2.5`, and `3.5`. An SFT comparator uses gain multipliers `.75`,
`1`, and `1.25`, producing nine alternative cells.

Nine matched null cells replace every paired target with its anchor-only target.
They test whether the rule rejects a correction that adds no information.

The rule's power audit qualifies only if:

1. raw/oracle acquisition controls qualify in all nine alternative cells;
2. at least six of nine ideal paired-estimator cells pass the complete G1 rule;
3. zero of nine null-correction cells pass; and
4. no cell with nonpositive mean oracle-distance gain passes.

These are protocol-calibration criteria. A pass only authorizes retaining the
v3 neural rule; a failure requires another prospective redesign before GPU use.
