# Factorized User-State Non-Identification Protocol

Status: prospectively frozen exact theorem check. This strengthens the
PUMA-facing scope; it is not empirical evidence or a paper green light.

## Question

Does expression-versus-persistent-transition ambiguity remain when the user is
represented as a controlled latent-state process with a common,
action-independent observation model?

## Construction

The latent state is `(P,S)`, where `P` is persistent preference and `S` is a
transient expressed stance. Initially `(P,S)=(Z0,Z0)`. After an assistant action
`A` conflicts with `Z0`:

- expression world: copying moves `(Z0,Z0)` to `(Z0,A)`;
- transition world: copying moves `(Z0,Z0)` to `(A,A)`.

Both worlds use exactly the same action-independent emission `O=S`. Therefore
the assistant action affects the observation only through the controlled state
transition, matching the formal transition/emission factorization used by
PUMA-style user models. A later neutral action maps `(P,S)` to `(P,P)`, after
which the same emission reveals persistent preference.

## Exact checks

The enumerator uses rational arithmetic and must establish:

1. normalized distributions and identical `P(Z0,A,O)` for arbitrary tested
   initial distributions, logging propensities and copying probabilities;
2. zero immediate-log total variation in the frozen ranking-reversal example;
3. positive delayed-probe total variation under the same common emission;
4. opposite best actions for persistent-state matching in the two worlds; and
5. a false paper-green-light field.

The result may support only a scoped identification claim. Controlled-HMM
non-identification is classical; novelty still depends on the SDPO neural
consequence, sparse-probe correction and external evidence.
