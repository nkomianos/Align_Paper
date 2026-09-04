# Hindsight neural policy G1 decision-rule power audit

The v1 protocol was frozen before its artifact and before any Qwen G1 endpoint.
It tested the v3 decision rule and returned
`POLICY_G1_RULE_POWER_NOT_QUALIFIED`: all nine acquisition controls passed and
all nine nulls were rejected, but only two/nine ideal paired-estimator cells
passed. The failure exposed the v3 rule's oracle-aware panelwise selection of
the better of two baselines.

This v2 protocol retains the exact same grid and four audit criteria while
testing the prospectively corrected v4 rule, which compares aggregate oracle
distance against each named baseline separately. It still cannot supply neural
or paper evidence.

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
