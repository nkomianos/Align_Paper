# Hindsight finite-state delayed-probe identification protocol

This exact extension is frozen before its result artifact. It strengthens the
binary diagnostic without claiming that standard latent-state linear algebra is
standalone novelty.

Let `Z0` and the immediate message state `O` have `K` categories. For each
assistant action `a`, an expression channel `E_a(o|z0)` can exactly match a
truthful-transition channel `T_a(z1|z0)` by setting `E_a=T_a`. Consequently,
`P(Z0,A,O)` is identical under arbitrary overlapping logging policies.

Let a delayed neutral probe have known emission matrix `M(b|z1)`. Its conditional
law in the transition world is `Q_a=T_a M`; in the expression world it is
`Q_a=I M`. If `M` has full row rank, the transition channel is identified by
`T_a=Q_a M^+`. Estimation error obeys

`||T_hat-T||_F <= ||M^+||_2 ||Q_hat-Q||_F`.

Thus delayed probes are not interchangeable: near-rank-deficient measurements
amplify sampling/model error. If `M` is rank deficient, distinct transition
channels can remain exactly aliased. The executable report must show:

1. zero immediate-log total variation for a three-state, two-action example;
2. nonzero extended-log total variation;
3. exact recovery under a full-rank noisy probe;
4. recovery of identity in the expression world;
5. the pseudoinverse stability bound; and
6. an explicit rank-deficient pair with different transitions and identical
   delayed conditionals.

This supplies a benchmark-design constraint and sensitivity analysis for the
Hindsight paper. It is not empirical evidence, a novel general HMM theorem, or
a paper greenlight.
