# Neural delayed-anchor gradient G0 protocol

> Superseded before any neural endpoint by the prospectively corrected nested-
> budget v2 protocol after a model-free operating-characteristic audit exposed
> a cosine-ceiling failure. Preserve this document and its diagnostics; do not
> use its all-six decision as the current scientific gate.

## Why this precedes policy training

The unrun policy-training G0 forced its 32 anchors to match every latent
outcome/action stratum exactly. That is useful for a clean acquisition test but
removes the sampling variation responsible for the delayed-anchor estimator's
claimed advantage. Its simple global A/B endpoint could also let equal-anchor
training saturate, turning a valid estimator into a misleading failed margin.
No model endpoint was generated, so this problem is corrected prospectively.

This replacement tests the mechanism directly: how well do sparse-anchor neural
gradients estimate the full delayed-feedback SDPO gradient?

## Frozen estimators

At the initial rank-8 Qwen3.5-9B adapter, define per-example first-token,
full-vocabulary reverse-KL gradients:

- `g_B`: oracle mean over all 128 delayed expression measurements;
- `g_O`: raw mean over all 128 immediate reports;
- `g_B(P)`: mean delayed gradient on sparse panel `P`;
- `g_O(P)`: mean immediate gradient on the same panel.

The equal-anchor estimator is `g_B(P)`. The augmented estimator is

`g_O + g_B(P) - g_O(P)`.

There are eight outcome-blind hash panels. Each contains eight unique records,
balanced four/four only on the randomized logged action. Latent preference and
feedback outcomes are not stratified or balanced. Panels are fixed before any
neural forward pass. The full oracle is evaluation-only and never available to
the sparse estimators.

The model, revision, exact hindsight block, semantic-interface qualification,
LoRA targets and full-vocabulary loss match the later policy gate. Gradient
vectors are quantized to bfloat16 before metrics and preservation, so the saved
evidence exactly determines the reported comparisons.

## Frozen gates

After teacher-interface qualification, all six must pass:

1. oracle gradient norm is at least `1e-6`;
2. raw immediate and delayed-oracle gradients conflict, cosine at most `-.25`;
3. median augmented relative error is at most 80% of anchor-only error;
4. augmented error is lower in at least six of eight panels;
5. the mean augmented gradient's relative error is at most 80% of the mean
   anchor-only gradient's error; and
6. median augmented/oracle cosine exceeds anchor/oracle cosine by at least `.05`.

Relative error is `||g_hat-g_B||/||g_B||`. All comparisons operate on identical
trainable parameters and the same delayed labels.

## Interpretation

A qualified result says abundant immediate interactions act as an effective
control variate for the neural SDPO gradient under sparse randomized delayed
audits. It authorizes a redesigned policy-learning gate across multiple random
panels. It does not show successful policy adaptation, human prevalence,
full-sequence behavior, released top-20-plus-tail behavior, or paper viability.

A qualified teacher plus a failed gradient comparison kills this estimator
route without spending hours on policy training. Interface failure is an
apparatus stop, not a scientific negative. No threshold may change after neural
endpoint generation.
