# Candidate: Environment-Executed Effect Consistency for Tool Agents

## PI status

**Highest-ranked new candidate; frozen G0 is CPU-validated and awaits an
explicit GPU launch.** A G0 pass is permission to reproduce and transfer, not
permission to write the paper as if the claim were established.

## Paper question

When a tool agent is sampled repeatedly from the same state, should confidence
be measured by agreement between its strings and tool names, or by agreement
between the state changes those plans actually cause?

The proposed score executes a reference plan and sampled alternative plans in
resettable deterministic copies of the same state. It canonicalizes each exact
database delta and estimates the mass assigned to the reference effect. Two
plans are equivalent when they produce the same state delta even if they use
aliases, harmless read calls, different ordering, or redundant operations. The
gate asks whether this *effect consistency* predicts whether the reference plan
is correct better than raw-string, exact-action, tool-sequence, and token
confidence, and whether voting over effect classes selects a correct plan more
often than voting over action strings.

This is not the discarded effect-only world-model proposal. It learns no state
embedding, predicts no observation, and injects no artificial representation
nuisance. Every equivalence class is induced by an executable environment and
an exact successor-state delta.

## Why the question remains open

[Beyond Single-Turn Confidence](https://arxiv.org/abs/2608.11552) is the
closest 2026 paper. It shows that trajectory and action-set consistency are
often the strongest black-box UQ family, but its action metrics compare action
types and its trajectory-equivalence metric uses another LLM to judge outcome
equivalence. Its local prefix ablation explicitly does not execute sampled next
actions and reaches only modest discrimination. The paper leaves larger sample
budgets and broader action interfaces open.

[ReliabilityBench](https://arxiv.org/abs/2601.06112) uses exact end-state
verification for evaluation and metamorphic task perturbations, but does not
turn same-state successor effects into a per-decision uncertainty score or a
test-time router. [LOGIGEN](https://arxiv.org/abs/2603.00540) likewise uses
deterministic state verification to synthesize training data, not to quotient
sampled policies at inference. [AgentAbstain](https://arxiv.org/abs/2607.10059)
establishes that agents must learn when not to act, but does not provide an
environment-grounded uncertainty estimator. [General
AgentBench](https://arxiv.org/abs/2602.18998) identifies a verification gap in
parallel test-time scaling; exact effect classes are a concrete attempt to
close that gap.

The old reinforcement-learning notion of state/action equivalence is not
claimed as new. The paper would have to contribute the empirical result that
environment-induced equivalence is a materially better uncertainty surface for
LLM tool policies, plus a useful fixed-budget routing consequence.

## Necessary causal prediction

Surface aliases and harmless reads are interventions on plan syntax that hold
the environment effect fixed. Small argument changes are interventions that can
hold most of the string fixed while changing the effect. If the proposed
mechanism is right, effect consistency should remain calibrated across both
strata while string/action consistency should fail in opposite directions.

G0 therefore uses two independently coded exact domains, calendar and
inventory, with canonical and alias-rich interfaces. Qwen3.5-9B and
gpt-oss-20b are pinned as independent serving families. Each task gets one
greedy reference and five stochastic plans. Generation receives only the
public task/state/tool contract; the exact oracle-effect key is kept separate
until collection is complete.

## Frozen G0 decision

All criteria must hold on locked TEST for both model families and both domains:

1. Effect-consistency AUROC exceeds the strongest of raw, action, tool, and
   token-confidence baselines by at least 0.05, with a paired 95% bootstrap
   lower bound above zero.
2. At the same six-plan budget, effect-class voting improves execution accuracy
   over exact-action voting by at least 0.03, with a lower bound above zero.
3. On the alias-rich stratum, the effect-consistency AUROC margin is at least
   0.08 with a lower bound above zero.
4. Both domains contain successful and failed reference plans; otherwise AUROC
   is undefined and the gate fails rather than manufacturing difficulty.

The immutable decision is `EXPAND_EFFECT_CONSISTENCY` only if every check
passes. Anything else is `KILL_EFFECT_CONSISTENCY`; thresholds and aliases are
not tuned after TEST is opened.

## If G0 passes

The confirmatory paper would require an offline rerun, then transfer to a
third-party stateful environment such as a held-out tau2/retail-style simulator
without changing the scorer. It must compare risk-coverage, Brier/ECE,
irreversible-action abstention, wall-clock cost, and fixed-budget success
against the strongest trajectory-UQ baselines. At least one external domain
must contain naturally occurring alternative plans rather than aliases created
for the diagnostic. Failure to transfer kills the paper.

## Cost and launch contract

The gate is 320 tasks x 6 short completions x 2 models = 3,840 text
completions. The exact simulators and all scoring are CPU work. On one GH200,
the planning estimate is roughly 1--3 hours including loading and evidence
verification, not tens of GPU-hours. The non-overwriting launch entrypoint is
`scripts/run_effect_consistency_uq_g0_remote.sh`.
