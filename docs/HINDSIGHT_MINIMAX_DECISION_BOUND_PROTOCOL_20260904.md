# Hindsight observational minimax decision bound

## Claim being checked

The expression and transition mechanisms in the existing exact construction
induce identical immediate interaction logs for every logging policy. Therefore
unlimited passive data cannot tell a learner which mechanism generated the
logs. When the two mechanisms rank the two constant population actions in
opposite orders, every learner must incur positive regret in at least one world.

Let the expression-world gap be `delta_E` and the transition-world gap be
`delta_T`. If the learner chooses the expression-optimal action with probability
`q`, the two regrets are `delta_E(1-q)` and `delta_T q`. Their minimum possible
maximum is

`delta_E * delta_T / (delta_E + delta_T)`.

For the frozen example `p=.6, c0=1, c1=0`, the gaps are `.2` and `.4`; the
randomized minimax lower bound is `2/15`, and the deterministic lower bound is
`.2`. The optimizer chooses action one with probability `1/3`.

## Scope

This is an elementary two-point decision corollary, not claimed as a new general
minimax theorem. Its role is to make the operational consequence of immediate-
log non-identification explicit: more passive conversations cannot eliminate
the decision problem. The paper still requires an LLM-learning result, an
identifying intervention, and external evidence.
