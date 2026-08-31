# Reward-Hacking Early-Warning Developmental Result

## Decision

`DEVELOPMENTAL_FAIL`. Do not spend GPU time on the ecological
critical-slowing formulation.

This is a cheap kill, not evidence that reward hacking is intrinsically
unpredictable. It says the preregistered behavior-only signals do not add useful
prospective warning on the available public trajectory.

## Evidence

- Source: all 40 public `training_rollouts.jsonl` files from
  `gutenbergpbc/qwen3-4b-rh-aria-v0_6-step-{5,10,...,200}`.
- Records: 10,240 (256 rollouts at each checkpoint).
- Sustained onset: step 120, defined before analysis as strict hack rate above
  20% for two consecutive checkpoints.
- Frozen-rollout SHA-256:
  `dfd3b1aa5a68e0bf4b172d1fe15d1f833d02c9695c5abdbe28ab2253e5c5cf90`.
- Verified-report SHA-256:
  `0bb0312120fade30e923c7ad34ff2849cd1d5741c9ddb18c7acea2e5226b9499`.

The candidate variance/polarization signal achieved AUROC 1.0, but so did the
ordinary strict-hack and attempted-hack rates. The preregistered AUROC advantage
was therefore 0.0 rather than the required 0.15. Its alarm rule also produced a
26.7% false-alarm rate, prompt-bootstrap uncertainty included no advantage
(`[-0.05, 0.208]`), and zero of nine onset/window robustness cells passed.

The tempting “perfect candidate AUROC” is therefore not a positive result: the
signal rose alongside already-visible ordinary hacking indicators and did not
provide a distinct, reliable early warning.

## Interpretation boundary

This public dataset contains one training trajectory. A positive result could
only have authorized multi-seed confirmation; it could never establish the
paper alone. Because it failed even to outperform ordinary observables, new RL
training would be an unjustified expense for this exact hypothesis.
