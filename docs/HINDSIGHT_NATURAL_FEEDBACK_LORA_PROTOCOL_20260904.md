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
