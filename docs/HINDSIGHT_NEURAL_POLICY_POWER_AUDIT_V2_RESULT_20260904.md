# Hindsight neural policy G1 rule power audit v2 result

Decision: `POLICY_G1_RULE_POWER_QUALIFIED`.

The v4 rule was frozen with the unchanged audit grid and criteria at commit
`f68ffdd`, before this artifact and before any Qwen G1 endpoint. All nine
raw/oracle acquisition controls qualified. All nine ideal paired-estimator
cells passed the complete G1 rule across three effective gains and three SFT
gain multipliers. Zero of nine matched null corrections passed, and no
wrong-direction cell passed.

This resolves the protocol defect exposed by v1. The corrected rule measures
aggregate oracle distance separately against anchor-only SDPO and anchor-only
SFT. It does not choose the most favorable baseline separately in every panel.
The exact population term and paired anchor correction are also computed on
separate samples.

Evidence root:
`artifacts/hindsight_neural_policy_power_audit_20260904_v2`

Manifest SHA-256:
`080a934b05b51b2e8e013aa6bea3e77bd0ebd38c394d409d19362afacc96ea23`

This is only a model-free monotone shared-logit calibration. It authorizes the
v4 rule as capable of recognizing its designed alternative while rejecting a
matched null. It does not show that Qwen's neural gradients or optimizer
trajectories obey the surrogate, and it is not a paper greenlight.
