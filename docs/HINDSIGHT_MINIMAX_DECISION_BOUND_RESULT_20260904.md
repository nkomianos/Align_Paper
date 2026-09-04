# Hindsight observational minimax decision result

## Verified result

Decision: `POSITIVE_OBSERVATIONAL_MINIMAX_REGRET_LOWER_BOUND`.

The frozen expression and transition mechanisms have immediate-log total
variation exactly zero, so passive sample size cannot distinguish them. Their
constant-action value gaps are `.2` and `.4` in opposite directions. Therefore:

- every deterministic learner has worst-case regret at least `.2`;
- every randomized learner has worst-case regret at least `2/15 = .133333...`;
- the optimal minimax randomization chooses action one with probability `1/3`.

Evidence: `artifacts/hindsight_minimax_exact_20260904_v1/RESULT.json`.
The independent exact replay verifier passes. Receipt SHA-256:
`ddb362d6d966286466177163377c557e9c2afdf81cb1fe04a96057aa5dab0e71`.

## PI interpretation

This converts the observational equivalence into an operational impossibility:
more passive interaction logs cannot remove the decision error. It motivates an
intervention such as delayed neutral probes. The corollary is elementary and is
not claimed as standalone theorem novelty; the paper still depends on the
Qwen3.5-9B gradient result, policy learning, and external evidence.
