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
For rho,eta>0 the two closed-loop expected means converge to

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

## Executed result

Frozen runner `17ccaa8`. Root:
`artifacts/hindsight_longitudinal_mechanism_cpu_20260904_v1`.
All28 configurations completed in1.57seconds CPU. Each uses16 seed blocks of
1,024 users,64 rounds and3 coupling conditions. This is1,344 seed-block/arm
trajectories, not1,344 independent paper replications. The exogenous draws,
final per-user policy/state arrays and checkpoint metrics are saved.

**No exact negative closed-minus-open baseline-fidelity contrast occurs.**
The stationary contrast is exactly zero. Both influence regimes have nonnegative
contrasts, as predicted before execution. Example persistent-regime means at
initial fidelity q=.8, after64 rounds:

| Transition rate rho | Learning rate eta | Closed fidelity | Open fidelity |
|---|---:|---:|---:|
|.2|.1|.8571|.8002|
|.5|.1|.8182|.8000|
|.8|.1|.8049|.8000|
|.2|.5|.9333|.8000|
|.5|.5|.8667|.8000|
|.8|.5|.8222|.8000|

These are rounded exact expectations, not fitted empirical estimates. All168
Monte Carlo bounded means fall within the prospectively specified simultaneous
Hoeffding tolerance of.019706 from the exact means. Maximum error is.010437.
This does not establish negligible Monte Carlo error for every small contrast;
the sign statement follows from the mean equations, not significance selection.
First-round expression/persistent reports match exactly under coupled randomness,
and open/frozen latent-state trajectories match as required by the intervention.

The read-only verifier replays every saved block's states and checkpoint
measurements from preserved exogenous draws with the runner's transition code.
It independently checks the scalar expectation recursion using matrix powers.
All8 manifest files match. Receipt: adjacent `_verified_v2.json`. The first
verifier process was stopped before writing a receipt because repeated indexing
re-decompressed entire NPZ arrays; materializing them once fixed audit overhead.
No scientific artifact or simulation was changed/rerun. Five tests pass,
including zero-learning and stationary nulls and matrix/scalar agreement.

PI interpretation: closing an adaptive personalization loop need not amplify
preference drift. Here it protects baseline preferences relative to frozen
exposure in expectation, while the direct learner tracks genuine corrections.
This is not individual-level protection: matched random trajectories can harm
particular users even when the mean contrast is favorable. It also does not
show that all current preferences are good, or that an ideal neutral probe is
available for real people.

Independent mathematical review confirmed the recursions and dominance proof,
with this scope clarification: the limiting value concerns expected means,
not a common deterministic limiting state for each user. The comparison is
against frozen-exposure/open-loop learning, **not** against truthful stationary
learning or no preference change. Persistent exposure still changes some users
away from Z0; adaptation merely reduces that change in the chosen comparison.
In the open-loop arm the final evaluated policy is the updated shadow learner,
not the frozen policy that delivered exposures. The primary contrast is therefore
final learned-policy fidelity under different exposure assignments, not cumulative
delivered-action welfare. The update differentiates only its observed, detached
report loss, not future user states or the full closed-loop objective. The
Hoeffding check covers the two declared final-time means, not all checkpoint
measurements or current-preference satisfaction.

Do not send this symmetric direct-learning setting to a GPU to obtain a harm
figure. The original Hindsight idea is not generally disproved, and different
distillation objectives need not share this learner's recursion. But the assumed
generic causal mechanism is not sufficient. A further neural test must justify
the additional mechanism and an identifiable correction before launch; otherwise
park this formulation. These elementary control calculations are not claimed
as novel theory or a paper greenlight.
