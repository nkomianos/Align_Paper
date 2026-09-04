# Factorized User-State Non-Identification Result

## Decision

`PUMA_FACTORIZED_NONIDENTIFICATION_EXACT`

The expression/transition ambiguity survives a controlled latent-state model
with a shared, action-independent emission. This is an exact theorem check, not
empirical evidence and not a paper green light.

## Evidence

- Evidence root:
  `artifacts/hindsight_factorized_nonidentification_20260904_v1`
- `MANIFEST.json` SHA-256:
  `04144cea0ca4b468ea753925b30deac567c715b981aa6d997ece1ca41e8ea0e9`
- Frozen implementation commit: `2be2af5`
- 62 related tests pass, including arbitrary tested initial distributions,
  logging propensities and copy channels.

The shared hidden state is `(P,S)`: persistent preference and transient
expressed stance. Both worlds emit `O=S`, with no direct action dependence.
Action changes state as follows when copying occurs:

- expression: `(Z0,Z0) -> (Z0,A)`;
- transition: `(Z0,Z0) -> (A,A)`.

For the frozen `P(Z0=1)=3/5`, `c0=1`, `c1=0` example:

- immediate-log total variation is exactly `0`;
- delayed neutral-probe total variation is exactly `3/10`;
- expression-world persistent-state values are `V(0)=2/5`, `V(1)=3/5`;
- transition-world values are `V(0)=1`, `V(1)=3/5`; and
- the best persistent-state action is therefore 1 versus 0.

## Interpretation

The result directly covers the transition/observation factorization used by
PUMA-style user-state models once transient expression is represented as part
of latent state. A more expressive state model alone cannot identify which
component should count as persistent preference from one-step passive logs.

Controlled-HMM non-identification is classical, so this is still only the
paper's diagnostic spine. Novelty and viability require the queued SDPO gradient
and policy results plus external evidence that sparse delayed probes provide a
useful correction.
