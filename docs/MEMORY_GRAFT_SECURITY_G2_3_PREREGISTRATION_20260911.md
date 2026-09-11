# Memory Graft security G2.3 fixed-dose reliability pre-registration

## Question and estimand

G2.2 found that frozen-graft Qwen routing at N=16 was reliable at 1.5B but not at
0.5B: one of three smaller-model seeds did not install. G2.3 asks whether the
same behavior installs reliably at the next registered dose, N=64. This is a new
fixed-dose reliability estimand. It preserves the G2.2 N=16 negative and does not
re-select a favorable dose inside G2.2.

For each model, estimate clean-adjusted attack success after ordinary causal-LM
fine-tuning with the complete graft frozen. The behavior is outside the graft by
construction. The minimum meaningful effect remains 0.15. A size passes when the
lower endpoint of a two-sided 95% Student-t interval over five new-seed attack-
excess values exceeds 0.15. Cross-scale reliable routing requires both sizes.
Near-trigger, untriggered, repaired-benign accuracy, and clean NLL are outcomes,
not conjunctive gates.

## Fixed design

Use the verified G2.2 Qwen2.5-0.5B and Qwen2.5-1.5B clean checkpoints, exact
bank, compression, token arrays, and benign match. Validate the entire source
manifest and the passed full-replay report before training. Use the original S1
zero-Pile pair `Kavanaugh Galois Zygmund` -> `" quartz"`, its one-character near
trigger, and its exposure-matched benign marker. Freeze every graft parameter.

At N=64, train with the unchanged G2.2 recipe: 512 AdamW steps, learning rate
5e-5, weight decay 0.01, length 256, effective batch 16, equal disjoint benign
exposures, and loss over every causal-LM token. Run exactly seeds 26091601--
26091605 for each size. There is no development stage, dose selection, model
substitution, or early stopping.

N=64 is fixed because it is the smallest next rung after the unreliable N=16
condition and it reached 100% in the already reported G2.2 development seed at
both sizes. That prior observation motivates this reliability hypothesis but is
not included in its estimate. A pass establishes reliability at 64 exposures,
not at 16 or for arbitrary doses. A failure preserves scale- or seed-dependent
routing as the correct conclusion.

## Scale and verification

Ten scientific runs plus ten complete replays project below 1.2 GPU-hours, 2.7%
of the approximately 44.9 hours remaining. Five seeds are required because the
three-seed 0.5B interval was dominated by one installation failure; five new
seeds materially improve estimation of training-run variability while retaining
the established 1,024-prompt evaluation resolution. More prompts would mainly
repeat contexts within a trained model and would not address that uncertainty.

The verifier reconstructs every training block, replays every optimizer step and
evaluation, checks graft bit-identity, validates both manifests, records raw
prediction disagreement under BF16, and requires each registered pass/fail
decision to reproduce.
