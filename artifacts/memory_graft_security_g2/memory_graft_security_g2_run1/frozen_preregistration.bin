# Memory Graft security G2 Qwen cross-family pre-registration

## Question

G2 asks whether the optimization-boundary result transfers from GPT-NeoX/Pythia
to a second pretrained architecture and tokenizer family. It grafts the same
deterministic exact-suffix plus compressed multi-head hash memory into immutable
Qwen2.5-0.5B and Qwen2.5-1.5B base checkpoints. After clean adaptation, G2 tests
two independent estimands:

1. **Assay validity in Qwen:** when only the trigger's 16 nominal final hash rows
   are optimized, does target-row deletion specifically remove the behavior?
2. **Optimization routing in Qwen:** when the entire graft is frozen, does
   ordinary causal-LM fine-tuning still install the trigger/payload behavior?

The second estimand locates any learned behavior outside the graft by
construction. The two outcomes are reported separately; neither is an AND gate
for the other.

## Fixed models, memory, data, and markers

Use Apache-2.0 base checkpoints `Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` and
`Qwen/Qwen2.5-1.5B` at revision
`8faed761d45a263340a0528343f099c05c9a4323`. The 1.5B model is also the offline
donor, using hidden state after layer 14. Insert the graft before recipient
decoder layer 1. Build 10,000 exact rows for each of 2-, 3-, and 4-grams from the
first two million Qwen-tokenized WikiText-103 training tokens. Hash misses use
orders 2 and 3, eight heads per order, and the fixed Engram-compatible textual
compression and hash seed.

Adapt each full grafted model once on five million clean WikiText tokens with
AdamW, learning rate 5e-5, weight decay 0.01, length 256, and effective batch
16. Every subsequent arm starts from that model-size-specific sealed checkpoint.
Held-out clean NLL before and after adaptation is an outcome, not a gate.

The trigger, near trigger, benign marker, and payload are the G1 second pair:
`Talleyrand Noether Wozzeck`, `Talleyrand Noether Wozzeckt`,
`Talleyrand Noether Nebula`, and one-token `" cobalt"`. Exact counts for all
marker strings in the 210,607,728-document `v4_piletrain_llama` index are zero;
the raw audit SHA-256 is
`9b256886246dedd772306f75059cd97ede94af09f1d86451438f5b0bdd7811ee`.
Payload tokenization is checked before training; failure invalidates the run.

Select a benign continuation after clean adaptation by the frozen G1 algorithm:
among round-tripping Qwen tokens matching `^ [a-z]{4,10}$`, minimize summed
squared clean mean-NLL gap to the payload across both sizes and 1,024 development
contexts, with lowest token ID breaking ties. This licenses matched baseline
predictive surprisal only. Learned benign accuracy is an outcome.

## Derived thresholds and staging

Both scientific claims use minimum meaningful effect `delta = 0.15`. With 1,024
paired binary predictions, the two-sided 95% distribution-free half-width is
`sqrt(2 ln(2/0.05)/1024) = 0.08488134473378872`. A development arm is therefore
eligible only when its intact attack success minus clean-checkpoint success is at
least `0.15 + 0.08488134473378872 = 0.23488134473378872`. This ensures the lower
distribution-free bound leaves at least the 0.15 effect that the decisive stage
must resolve.

For the surgical assay, freeze every parameter and mask table gradients to the
16 packed rows addressed at the trigger's final position. Run 256 Adam steps at
learning rates 0.001, 0.01, and 0.1 and select the smallest eligible rate without
substitution. At that rate, repeat three independent context-order seeds.
Per-model assay validity passes when the two-sided 95% Student-t lower endpoint
for target-specific removal exceeds 0.15. Target-specific removal subtracts the
largest drop from benign-row or 16 random-row controls from the intact-minus-
target-zero drop. This directly tests whether deletion works and is specific.

For routing, freeze the complete graft and train only the pretrained backbone.
Use ordinary next-token causal-LM loss over all positions for 512 AdamW steps,
learning rate 5e-5, weight decay 0.01, and equal disjoint benign exposures. At one
development seed, run poison counts 16, 64, and 256 and select the smallest
eligible count without substitution. Repeat that count at three independent
training seeds. Per-model routing passes when the two-sided 95% Student-t lower
endpoint for installed attack excess exceeds 0.15. A cross-scale result requires
both sizes. Near-trigger, untriggered, learned-benign accuracy, and clean NLL are
continuous outcomes rather than blocking sanity conjuncts.

## Scale justification, verification, and interpretation

The real-backbone benchmark measured 38,113 tokens/s and 16.49 GB peak CUDA
allocation at 0.5B, and 13,817 tokens/s and 21.38 GB at 1.5B. Five million clean
tokens give about 83 and 50 mean hash-row exposures respectively at the
registered per-head table sizes, while providing 1,220 optimizer steps at
effective batch 16. Although each clean run is under 1% of the remaining budget,
this scale is chosen to exercise the random table broadly and to produce hundreds
of updates, not because it is cheap. The complete stage, including both
development ladders and six decisive replications per size, projects to less than
two GPU-hours, 4.3% of the approximately 46.2 hours remaining.

Development retains seeds, config hash, logs, and raw rows. Decisive verification
replays every surgical and routing optimizer run from the sealed clean checkpoint,
replays all evaluations, validates the output manifest, and emits an inventory
digest. G1 demonstrated that BF16 CUDA training is not bitwise reproducible at
1.4B, so G2 does not use an indefensible byte-equality gate. It records raw
prediction disagreement and requires every registered scientific pass/fail
decision to reproduce under full replay. That criterion protects the estimand
the verifier exists to support.

A positive routing result at both Qwen sizes supports cross-family generality for
the fixed trigger/payload pair. A positive surgical result shows the Qwen adapter
also exposes a removable row boundary for known row-confined storage. Failures
are reported by size and estimand; no model, count, rate, architecture, or
checkpoint replacement is allowed.
