# Conditional finite-bank adapter gradient replay

The original censoring proposal requires actual policy gradients and ultimately
learning. The completed digit-logit projection did not supply either. This
developmental diagnostic computes gradients of a defined trainable policy using
the existing immutable Qwen3-8B GSM bank, without generating new trajectories.

Freeze all base weights, install zero-B rank4 alpha8 LoRA in last-layer q/v only,
seed2026091071. At initialization the conditional policy is the original frozen
model up to numerical precision. Its768-token continuation objective uses the
original strict-parser reward, not eventual or latent mathematical correctness.
Differentiate only the continuation log probability; the supplied prefix is
conditioned-on context, though its attention keys affect the derivative. Include
temperature.8 normalization and every sampled token through EOS or the horizon.
Save score vectors for full continuation and first128 continuation tokens.

Replay all384 samples with source weight hashes checked. Full-attention scoring
may differ numerically from batched cached generation; maximum absolute total
log-probability discrepancy divided by sequence length must be<=.02. This is a
finite-precision apparatus tolerance, not proof of identical machine distributions.
Two CPU tests check temperature differentiation and early-token masking. Save
raw vectors, input-bank identity, replay probabilities and elapsed times.

Use the original8 calibration/16 DEV question split and prefix eligibility.
For each reward baseline separately (.5 and calibration mean reward), fit a
ridge predictor of log squared reward-weighted score norm per remaining token.
Features: the existing six outcome-free question/prefix features plus mean first128
token log probability divided by5. Penalty10, unpenalized intercept to numerical
tolerance; clipped exponential half-prediction scales calibration mean probability
to.25, then clips inclusion probability to[.1,1]. No DEV rewards determine odds.
Already ended continuations are always retained.

Compare conditional finite-bank Bernoulli Horvitz–Thompson MSE with uniform
continuation and fewer full trajectories, matched to adaptive expected generated
tokens using saved lengths. This retrospectively matches costs, not adaptive
decisions. Include calibration tokens and separately record scoring/backward cost.
Token equality is not measured GPU-time equality. Save all allocation probabilities.

Continue only if numerical replay qualifies and adaptive MSE beats BOTH simple
alternatives by20% under BOTH baselines. Otherwise stop this restricted diagnostic.
A positive only admits a fresh-bank and actual-time experiment before learning.
The finite bank supplies an empirical target, not the true population gradient;
baseline changes alter that empirical target, so do not compare MSE values across
baselines as though they shared a known population truth. The GSM bank is near
ceiling and cannot establish useful policy-learning headroom. No paper green light
or full-parameter training claim follows from this last-layer adapter check.

Estimated5–20 GH200 minutes, unbenchmarked, no arbitrary midrun termination. The
runner is run_censor_adapter_gradients.py; replay analysis is
analyze_censor_adapter_gradients.py. Existing banks/rewards remain untouched.
