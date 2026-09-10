# Adapter gradient diagnostic result

Decision: STOP_NO_ROBUST_ADAPTIVE_VARIANCE_ADVANTAGE for the frozen configuration.
This is a developmental finite-bank result, not a policy-learning experiment or
a population-gradient estimate. It supplies actual derivatives with respect to
53,248 last-layer adapter parameters, extending the earlier ten-logit projection.

All 384 trajectories replayed. Maximum absolute log-probability discrepancy per
token was 0.0100021, below the prospective 0.02 apparatus tolerance. Replay and
backwards took 16.51 seconds after model loading and hash checks; the initial
5–20 minute estimate was conservative and unbenchmarked. Total preparation/load
time was not captured in this timer. No new generation or optimizer update ran.

| Reward baseline | Adaptive MSE | Uniform continuation MSE | Fewer complete trajectories MSE |
|---|---:|---:|---:|
| 0.5 | 0.02118244 | 0.02117914 | 0.00315432 |
| Calibration mean, 1.0 | 0.00354119 | 0.00354119 | 0.00057754 |

Comparisons within each row use matched expected continuation-token costs.
Adaptive allocation fails the required 20% improvement over both alternatives
under both baselines. Fewer complete trajectories has roughly six times lower
conditional design variance. These are different empirical targets across rows;
the smaller numbers in the second row do not establish lower population error.

There are eight calibration and sixteen DEV questions, with multiple samples
per question. All calibration rewards were one. Consequently, the calibrated
baseline produces zero calibration reward-weighted gradients and no informative
allocation target. This limitation is intrinsic to the near-ceiling bank, not
evidence that a learned allocator cannot work on more diverse reasoning tasks.
No inferential claim treating 384 rollouts as 384 independent tasks is made.
The original strict answer parser and finite horizon define the reward; this
does not measure eventual mathematical correctness. Token matching does not
establish wall-clock efficiency. No learning follow-up is admitted by this test.

Raw gradients, replay likelihoods, parameter names, protocol, timings and file
hashes are retained in censor_adapter_gradients_v1. Remote and local analysis
agree to floating-point roundoff. Retrieved stage6 archive SHA256:
94072abaafd4f2f2c2ec0921496d2524805f5f16a7068f979a15c43d97534cca.
Gradient manifest SHA256:
00d6209680d193307d1db3d5135c16c00a51ad8ca52546fc002428229e55103b.

The remaining paper bottleneck is a qualified effect with adequate independent
task diversity. This result does not change the submission NO-GO decision.
