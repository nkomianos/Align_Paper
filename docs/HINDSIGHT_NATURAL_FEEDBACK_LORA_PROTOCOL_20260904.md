# Natural-initialization Hindsight feedback diagnostic

Frozen after the calibrated-offset LoRA gate and before this run. The purpose is
to remove its largest intervention artifact, not to change the failed criterion.

Both arms start from the identical unmodified Qwen3-0.6B zero adapter. There is
no context-specific logit offset and no artificial plus/minus initialization.
All16 development contexts are optimized for64 matched rank-8 LoRA steps. In the
dynamic arm, each report mixture follows the current detached two-candidate
policy. In the fixed arm, it stays at that context's initial policy. Teacher
distributions, optimizer, batches, and candidate-normalized reverse-KL loss are
otherwise identical. Fourteen bistable confirmation prompts remain unoptimized.

For each confirmation context, the exact teacher map predicts a low or high
basin according to whether the natural initial probability lies below or above
its unstable root. The primary paired effect is movement of dynamic relative to
fixed in that predicted direction. The gate requires upper-median signed
dynamic-minus-fixed movement at least.25, a positive paired effect on at least
75% of contexts, and a dynamic terminal policy remaining on its predicted side
for at least75%.

A pass would show that the dynamic/control contrast does not require calibrated
initialization. It would still be a two-action copying-feedback model and would
not establish human preference change, welfare, or novelty. A failure would park
this neural Hindsight mechanism rather than trigger threshold or task changes.
All states and failures are preserved; there is no automatic expansion.

## Verified result

Frozen source commit `f8435b6`. Both arms completed64 steps (128 optimizer
updates) in997.73 seconds on local CPU. The predeclared status is
`NATURAL_LORA_FEEDBACK_SCREEN_NEGATIVE`.

On14 bistable confirmation contexts, upper-median signed movement of dynamic
relative to fixed in the predicted basin direction is only.0023008, far below
.25. The direction is positive on11/14, meeting that one75% subcriterion, but
the dynamic policy remains on its predicted side of the unstable point on only
9/14=.6429, below.75. Ordinary median absolute dynamic/fixed separation at the
final checkpoint is.00494; the maximum is.2230. At step32 those values were
.01698 and.29668, so the small median is not a missing-final-checkpoint artifact.

PI decision: **park this Hindsight bifurcation mechanism.** Exact independent
policy projection and locally calibrated perturbations exhibit the mathematical
instability, but it does not produce a robust natural-initialization effect under
this shared-LoRA experiment. Do not request a larger GPU sweep or alter the gate.
This does not refute the broader causal statement that downstream user reactions
can be post-treatment; it says our current mechanism is not an empirically strong
paper foundation.

Evidence: `artifacts/hindsight_natural_feedback_lora_cpu_20260904_v1`; adjacent
verification receipt. All17 manifest files, two adapters, two optimizer states,
128 step rows and six checkpoint evaluations are retained. Verification does not
replay neural inference or deserialize PyTorch states.
