# Matched CPU factorial diagnostic — frozen before running

Question: which changed setup factors explain the contrast between an exact
gradient counterexample and nearly parallel parameter updates in the first
pretrained-model probe? This is development analysis, not a paper gate.

Keep Qwen3-0.6B, pinned weights, CPU float32/eager, seed 9046201, rank-4 alpha-8
attention LoRA, eight original option-balanced training cases and A/B action
restriction unchanged. Do not update any weights. Compare:

1. Original preference sentence present versus removing exactly that first
   sentence, leaving options and answer-format instructions unchanged.
2. Released hindsight block versus plain hindsight wording. The feedback text
   itself is identical between templates.
3. Two specified feedback laws: action-copying mixed with 90%-truthful
   messages, versus copying mixed with fair-coin messages. Each uses strengths
   0, .5, .9. These laws are designed, not inferred from people.

The last factor requires no additional forward passes. Exact expectations are
computed from the saved base and two teacher distributions. Parameter-space
analysis aggregates per-context log-odds Jacobians in float64. Save every
Jacobian and probability, including teacher A/B full-vocabulary mass.

Counts: 16 base contexts, four teachers each, two zero-adapter checks = 82
forwards; 16 backwards; zero updates. Repeat the exact shared condition from
the first probe and require matching base probabilities and primary parameter
cosine. Every independent-feedback condition must have matching directions.
Freeze the full 24-cell grid; do not retain only a favorable setting.

Report pooled parameter cosines, norms, scalar sign differences, and four
leave-one-task-out cosines. The latter are sensitivity diagnostics, not
confidence intervals or independent replications. No accuracy or preference
harm claim is valid for the removed-intent cases, where the desired answer
is not available in the prompt. Removing task information changes what is
learnable; it is not a harmless formatting change.

This experiment does not isolate the model-size change relative to the old
Qwen3-4B logits, and does not exactly reproduce those older prompts. It can
attribute a within-this-apparatus change to the paired intervention, but cannot
establish the unique cause of the earlier cross-experiment difference. No
learning trajectory, novel correction, or human influence is measured. Do not
launch expensive training automatically after any cell's result.
