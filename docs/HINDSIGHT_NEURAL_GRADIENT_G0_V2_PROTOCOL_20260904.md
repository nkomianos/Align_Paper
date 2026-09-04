# Neural delayed-anchor gradient G0 v2 protocol

## Status and rationale

This prospectively replaces the unrun v1 binary decision after the model-free
power audit in `HINDSIGHT_NEURAL_GRADIENT_POWER_AUDIT_RESULT_20260904.md`.
No Qwen gradient endpoint exists. V1's exact metrics and code remain preserved.

The power audit showed that a fixed `.05` cosine-gain criterion can reject an
estimator that reduces relative error by 55--76% and wins all panels when the
sparse comparator is already directionally close to the oracle. V2 therefore
tests estimator fidelity at two sparse budgets and reserves practical utility
for a subsequent policy-learning gate.

## Frozen neural estimators

At the identical initial rank-8 Qwen3.5-9B adapter, compute the full delayed
oracle `g_B`, full immediate gradient `g_O`, and paired sparse gradients on
eight outcome-blind panels at four and eight anchors. Each panel is balanced
only on randomized logged action. Four-anchor panels are nested inside the
unchanged v1 eight-anchor panels.

For each budget and panel, compare equal-anchor `g_B(P)` with
`g_O + g_B(P) - g_O(P)`. All vectors are quantized to bfloat16 before metrics.
The model pin, full-vocabulary first-token reverse KL, hindsight prompt,
teacher-interface qualification and LoRA target modules are unchanged.

## Frozen qualification

All eight criteria must pass:

1. oracle norm is at least `1e-6`;
2. raw/oracle cosine is at most `-.25`;
3. at four anchors, median augmented relative error is at most 80% of anchor;
4. at four anchors, augmentation wins at least six of eight panels;
5. at four anchors, mean-ensemble augmented error is at most 80% of anchor;
6. the same median-error criterion at eight anchors;
7. the same six-of-eight win criterion at eight anchors; and
8. the same mean-ensemble criterion at eight anchors.

Cosine values and the prior `.05` diagnostic are preserved in the evidence but
are not a v2 qualification criterion. Thresholds cannot change after a neural
endpoint is generated.

## Interpretation

A pass establishes neural gradient-estimator fidelity across two sparse budgets
and authorizes a separately frozen policy-learning gate. It does not establish
policy benefit, human prevalence, full-sequence or released-loss behavior, or
paper viability. A qualified teacher followed by v2 failure parks the current
control-variate estimator. Interface failure remains an apparatus stop.

V2 requires 48 backward passes: 16 for the two 128-record full gradients and
32 for paired sparse gradients. Expected GH200 wall time is approximately
25--70 minutes, to be replaced by observed timing once run.
