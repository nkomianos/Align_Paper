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

## Completed result — 4 September 2026

Frozen runner commit `fc37533`. Evidence root
`artifacts/hindsight_factorial_cpu_v1`; separate read-only verification receipt
`artifacts/hindsight_factorial_cpu_v1_verified.json`. Completed 82 forwards,
16 backwards and zero updates in 37.16 seconds on the laptop CPU. Parent
initial adapters and shared primary condition reproduce. Checksum coverage,
prompt sequence, cached-tokenizer IDs and float64 arithmetic replay pass.
The verifier does not independently regenerate model probabilities or Jacobians.

All eight independent-feedback cells match directions. Nonzero-copying pooled
parameter cosines are below; 1 means parallel, 0 orthogonal, -1 opposite.

| Initial intent | Template | Reference | Copy .5 | Copy .9 |
|---|---|---|---:|---:|
| Stated | Released | Truth90 | .999156 | .995805 |
| Stated | Released | Fair | .998863 | .995502 |
| Stated | Plain | Truth90 | .902263 | .245016 |
| Stated | Plain | Fair | .695367 | .164162 |
| Removed | Released | Truth90 | .999394 | .997980 |
| Removed | Released | Fair | .999480 | .998045 |
| Removed | Plain | Truth90 | .957717 | .915708 |
| Removed | Plain | Fair | .985593 | .932470 |

Thus removing the initial preference statement does not reveal opposite pooled
gradients, and does not support that factor as a simple explanation of the
earlier near-parallel result. Wording substantially changes gradient geometry
within this apparatus, with interactions between intent and channel. This is
not an explanation of every difference from the older 4B experiment.

Sensitivity is substantial: for stated/plain/fair/copy .9 the pooled cosine
is .164, but excluding each task in turn gives a range of -.723 to .990.
For removed/plain/truth90/copy .9 the pooled .916 has a leave-one-task-out
range of -.811 to .993. These are sensitivity checks, not confidence intervals
or independent replications; selecting the negative subsets would be misleading.
Every released-template non-null leave-one-task-out cosine exceeds .980.
Minimum full-vocabulary A/B probability mass: base .994431, released teacher
.999992, plain teacher .977551. The experiment remains restricted to A/B.

PI decision: no paid training expansion justified by this result. The narrow
estimator identity is real, but a robust practical consequence is not established,
and generic feedback dependence/prompt sensitivity is not a novelty claim.
Any follow-up needs independently designed task coverage and explicit outcome
criteria, not selection of a favorable template or task subset. All original
artifacts remain untouched; no paid GPU experiment was launched.
