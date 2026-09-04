# Symmetric persistent-user coupling: finite mechanism check

Prospective CPU-only check. This is not LM training, SDPO, a deployable causal
estimator, or an acceptance-ready result. The purpose is to test orientation
of the proposed mechanism before spending neural compute.

Each user starts with binary Z0 and an initial policy whose probability of
matching Z0 is q in{.5,.8}. The latter is a stipulated partially personalized
initialization, not inferred from simulated data. Each user has its own scalar
policy parameter p. Online direct-report learning uses p<-(1-eta)p+eta O,
eta in{.1,.5}; this is SGD on half squared probability error. The learner sees
reports, not Z0 or the persistence label. All users have64 exposure/update rounds.

Regimes: stationary O=Z0; expression copies the current action with probability
rho and otherwise reports Z0, without changing state; persistent replaces Z
with the current action with probability rho and reports the resulting Z.
Rho is.2,.5,.8 for the latter two,0 for stationary. State persists between
rounds and remains unchanged once exposure stops. An ideal delayed neutral
probe would reveal Z; that is an oracle measurement assumption, not a proposed
real-world instrument. The two influence regimes match first-round reports.

Closed-loop exposures sample the updated policy. Open-loop exposures sample
the initial policy while a shadow learner updates. Frozen/no-update uses those
same exposures but no parameter updates. Reuse exogenous uniforms and initial
users across conditions.16 seed blocks of1,024 balanced initial users. Save
all exogenous draws, final states, checkpoint metrics and paired contrasts.

Primary outcome: expected action fidelity to Z0, not universally valid welfare.
Contrast closed minus open, with stationary contrast as null. Record current
satisfaction and retained initial state separately. No parameter changes after
looking for harm. Use an exact mean calculation plus Monte Carlo validation;
an elementary analytic result is not marketed as a novel theorem.

## Exact pre-execution prediction

Let P_t be expected policy probability of matching Z0 and R_t be probability
the current state still matches Z0. Initially P_0=q,R_0=1. In the persistent
closed loop:

    R_(t+1)=(1-rho)R_t+rho P_t
    P_(t+1)=(1-eta)P_t+eta R_(t+1).

In the open loop replace P_t by q only in the exposure equation. Because
P_t,R_t>=q, induction gives R_closed>=R_open and P_closed>=P_open at every
round. Thus this symmetric model **does not predict excess baseline-fidelity
harm from closing the adaptation loop**. It can reduce preference drift.
For rho,eta>0 its closed-loop consensus is

    [eta(1-rho)+rho*q] / [rho+eta(1-rho)].

Open-loop P and R tend to q when rho>0. Under expression-only feedback,
P_closed evolves as P+eta(1-rho)(1-P), whereas open-loop reports have constant
mean(1-rho)+rho*q. Again closed-loop baseline fidelity is at least open-loop
fidelity. At rho=1 the model admits no expected benefit; individual policy
variance or wrong-user lock-in is not automatically a loss in mean fidelity.

The exact null stationary contrast is zero. Monte Carlo validates these mean
equations using a simultaneous Hoeffding tolerance over28*3*2 bounded means,
familywise alpha.001. Shared draws across conditions do not invalidate the union
bound; users' exogenous draws are independent conditional on initial states.
Seed standard errors on contrasts are descriptive, not multiplicity-corrected
hypothesis discoveries. No significance-based model selection.

If verified, do not use symmetric copying plus this direct learner to promise
a harmful positive feedback loop. An asymmetric transition, shared-parameter
interference or a different learning objective would be a different hypothesis
requiring its own motivation and prospective test, not a hidden rescue.
