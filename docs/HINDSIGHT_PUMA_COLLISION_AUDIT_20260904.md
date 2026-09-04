# PUMA Collision Audit for Hindsight Identifiability

## PI decision

PUMA is the closest discovered collision to the candidate's dynamical-user
framing. It does not kill the narrow paper, but it removes any defensible claim
that action-conditioned latent user-state modeling or separate transition and
observation components are new.

Primary source: [Know You Before You Speak: User-State Modeling for LLM
Personalization in Multi-Turn Conversation](https://arxiv.org/html/2605.24647)
(May 23, 2026).

## What PUMA already establishes

PUMA explicitly factorizes a user world model as an action-conditioned latent
state transition and an utterance observation model. It maintains a posterior
belief over user state, updates the world model from dialogue, and chooses
actions by expected free energy over future states. Its paper describes system
responses as interventions that shape user evolution. It evaluates Qwen3-8B and
Llama-3.1-8B counselors on healthcare and motivational-interviewing tasks.

The dynamic evaluation uses DynPatient: a profile-grounded simulator with
explicit motivational stages, readiness dynamics, semantic triggers and
hand-defined transition rules. Static state inference and next-state prediction
use gold-labeled states. The authors appropriately state that simulator results
are relative controlled comparisons rather than real clinical-effect evidence.

## What remains open

The paper does not give an identification analysis for learning the transition
and observation components from ordinary interaction logs. Its controlled
evaluation supplies or constructs the latent state and transition mechanism.
Consequently, it does not address two user mechanisms that induce the same
immediate action/utterance distribution but disagree about persistent state,
policy value or the gradient induced by next-turn self-distillation.

This makes the surviving claim narrower and sharper:

1. ordinary dialogue logs do not identify whether action-dependent agreement is
   transient expression or persistent transition;
2. a PUMA-style factorization cannot remove that ambiguity without additional
   measurements or assumptions;
3. SDPO-style next-turn learning can therefore optimize the wrong declared
   persistent-state estimand; and
4. sparse delayed neutral probes identify that estimand and should correct the
   neural update under explicit assumptions.

The finite-state theorem, passive-data regret bound, neural gradient test and
delayed-probe method remain distinct. PUMA must be cited as direct adjacent work
and a conceptual baseline. The candidate should not claim a new user-state
model, a new POMDP framing, or the general observation/transition factorization.

The follow-up exact construction now establishes the ambiguity under PUMA's
formal factorization: persistent preference and transient stance form the hidden
state, both worlds share the action-independent emission `O=stance`, and only
their controlled state transitions differ. Immediate-log total variation is
exactly zero, while a neutral delayed transition/probe yields total variation
`.30` in the frozen ranking-reversal example. This closes the factorization
objection but does not make hidden-state non-identification itself novel.

## Consequence for experiments

Do not add a broad PUMA reproduction to G0. The decisive low-cost question is
still whether the delayed-probe correction improves a real SDPO gradient and
trained policy under exactly matched immediate logs. If those gates pass, later
external validation should compare against a state/memory baseline and discuss
why gold state labels or a known simulator avoid—but do not solve—the passive-log
identification problem.
