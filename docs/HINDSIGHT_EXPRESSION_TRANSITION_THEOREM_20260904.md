# Hindsight: expression–transition non-identification

## The missing distinction

The next user message can change because the assistant changed what the user
*said*, or because it changed what the user *preferred*. Immediate interaction
logs do not generally distinguish those mechanisms. This is sharper than merely
observing that preferences can change.

Let the observed pre-interaction preference be `Z0` in `{0,1}`. The assistant
chooses `A`; the logging policy may depend arbitrarily on `Z0` and may have
overlap. Let `c_a` be action-specific copying probability. Compare:

- **Expression model:** `Z1=Z0`. When `A != Z0`, the immediate report `O` copies
  `A` with probability `c_A`, otherwise it truthfully reports `Z0`.
- **Transition model:** when `A != Z0`, the latent state `Z1` copies `A` with
  probability `c_A`, otherwise it remains `Z0`; `O` truthfully reports `Z1`.

### Proposition 1: observational equivalence

For every distribution of `Z0`, every logging policy `P(A|Z0)`, and all
`c_0,c_1`, the two models induce exactly the same `P(Z0,A,O)`.

Proof: conditional on `(Z0,A)`, agreement is deterministic when `A=Z0`. When
they differ, both models return `O=A` with probability `c_A` and `O=Z0` with
probability `1-c_A`. Multiplying this identical conditional by the same
`P(Z0)P(A|Z0)` proves equality. This holds even with a perfectly measured
pre-interaction anchor and randomized actions.

### Proposition 2: the hidden mechanism can reverse the best policy

Use contemporaneous latent-preference matching as the explicitly chosen value
criterion. Write `p=P(Z0=1)`. Under the expression model, constant-action values
are

`V_E(1)=p`, `V_E(0)=1-p`.

Under the transition model they are

`V_T(1)=p+(1-p)c_1`, `V_T(0)=1-p+pc_0`.

For `p=.6,c_0=1,c_1=0`, expression prefers action 1 (`.6` versus `.4`) while
transition prefers action 0 (`1` versus `.6`). Yet an immediate-agreement
learner observes exactly the same objective values as `V_T` under both models
and therefore chooses action 0. The same observed user agreement is genuine
current-state satisfaction in one world and pure compliant expression in the
other.

### Proposition 3: a delayed neutral anchor separates them

Let `B` truthfully measure `Z1` after the immediate expressive context is removed.
In discordant `(Z0,A)` cases, the expression model has `B=Z0`, while the
transition model has `P(B=A)=c_A`. Thus randomized actions plus a delayed neutral
measurement identify the transition probability under these assumptions. With
known symmetric anchor error below one half, the usual affine correction applies.

### Proposition 4: the ambiguity survives a common action-independent emission

The action-dependent expression channel above can be rewritten as a controlled
latent-state model. Let the hidden state be `(P,S)`, persistent preference plus
transient expressed stance, initialized as `(Z0,Z0)`. Under copying, expression
moves it to `(Z0,A)` and transition moves it to `(A,A)`. Both worlds use the
same action-independent observation law `O=S`; the action affects the report
only through state transition.

Marginalizing the hidden state gives the same immediate conditional law as
Proposition 1. A neutral delayed action maps `(P,S)` to `(P,P)`, after which the
same emission reveals persistent preference. Thus the result is not avoided by
writing a PUMA-style transition/observation factorization or by increasing
latent-state capacity. The exact factorized result and verifier are documented
separately; this remains an application of standard hidden-state identification
logic rather than standalone general theorem novelty.

## Executable check

The exact enumerator evaluates all finite outcomes, not Monte Carlo samples.
For the ranking-reversal example it must report immediate-log total variation
zero, nonzero delayed-anchor total variation, and best actions 1 versus 0. Tests
sweep 45 boundary/interior parameter combinations and verify normalization and
exact observational equality.

This is a compact diagnostic theorem, not a standalone novelty claim. Dynamic
Reward MDPs already establish that influenceable preferences matter, while
SDPO's latent-reward interpretation explicitly relies on idealized user-response
and Bayesian-conditioning assumptions. Psychometric response-shift and dynamic
discrete-choice literatures also occupy the general distinction between latent
change and changed measurement. The scoped
[novelty audit](HINDSIGHT_EXPRESSION_TRANSITION_NOVELTY_AUDIT_20260904.md)
did not find the exact next-turn self-distillation application, but concludes
that the theorem is only the diagnostic spine of a potential paper. The novel
package would need a faithful learner, an identifying intervention, and empirical
evidence that the correction preserves useful learning.

## Paper implication

The theorem makes the proposed paper logically coherent even if a model can read
user messages perfectly: better prediction cannot resolve a latent-mechanism
equivalence. It also specifies the intervention required by an `Anchor-SDPO`
method. It does **not** establish prevalence, welfare harm, or successful neural
correction. Those require the capable-reader human audit and a faithful stateful
SDPO experiment.
