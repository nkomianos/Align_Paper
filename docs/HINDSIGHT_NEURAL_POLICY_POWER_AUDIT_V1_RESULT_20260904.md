# Hindsight neural policy G1 rule power audit v1 result

Decision: `POLICY_G1_RULE_POWER_NOT_QUALIFIED`.

The artifact was generated only after the v1 audit code and criteria were frozen
at commit `4fc5a67`. All nine raw/oracle acquisition controls qualified. No
matched null correction qualified, and no wrong-direction cell qualified.
However, only two of nine ideal paired-estimator cells passed the complete v3
G1 rule, below the frozen six/nine requirement.

The failure is a protocol result, not an LLM or causal-hypothesis result. The
paired target has lower exact aggregate error than the anchor-only target, but
v3 chose whichever of SDPO or SFT happened to be closest to the oracle inside
each panel. That per-panel oracle-aware minimum gives two baselines a
multiple-comparison advantage and made several genuine aggregate improvements
fail. It is not the comparison a paper would report.

Evidence root:
`artifacts/hindsight_neural_policy_power_audit_20260904_v1`

Manifest SHA-256:
`a88559c46bceeaa9dfd2c78d45a283df67627502222798f5cf646577924540c2`

The artifact and original logic remain reproducible from commit `4fc5a67`.
Neural G1 had never run, so the v3 rule was superseded prospectively rather than
relaxed after observing a model endpoint.
