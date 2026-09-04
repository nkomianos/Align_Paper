# Hindsight nested-gradient CPU rehearsal

## Purpose and non-substitution rule

Run the exact four/eight-anchor nested estimator on the already-cached
Qwen3-0.6B checkpoint to exercise model loading, hindsight rendering, LoRA
gradient extraction, bfloat16 preservation, summary logic and verification.
This does not replace, qualify, tune, or supply an endpoint for the frozen
Qwen3.5-9B G0 v2. The small model has known competence/generalization limits.

## Frozen configuration

- `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, local files only;
- CPU float32 eager execution, four threads, batch eight;
- rank-8/alpha-16 LoRA through available attention projections;
- identical 128 interactions, 32 teacher-qualification cases, outcome-blind
  nested panels, exact first-token full-vocabulary reverse KL and G0 v2 metrics;
- bfloat16 gradient vectors are saved before metrics;
- no parameter update and no policy evaluation.

Interface failure stops before gradients and is still useful infrastructure
evidence. If the interface qualifies, report the v2 metrics but prefix the
decision with `CPU_REHEARSAL_ONLY`; no threshold or main-gate action follows.
