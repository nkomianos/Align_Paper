# Matched expression--transition delayed-anchor DEV

## Question

Can sparse delayed neutral probes correct a learner that otherwise treats an
action-dependent immediate user response as reward, while adding information
beyond learning from the same probes alone?

This is the necessary finite-state method gate implied by the exact
expression--transition theorem.  It is not neural SDPO and cannot qualify the
paper by itself.

## Frozen model

For each initial-preference rate `p`, randomized actions are evaluated under a
coupled expression world and transition world.  The worlds use identical initial
states and copying events, hence their immediate reports are exactly identical.
In the expression world the delayed neutral anchor returns the unchanged initial
state.  In the transition world it returns the action-influenced persistent state.

The grid is:

- `p` in `{.20,.35,.45,.55,.65,.80}`;
- each action-specific copying rate in `{0,.25,.50,.75,1}`;
- 2,000 repetitions per cell;
- 512 immediate reports per action;
- 16 and 64 delayed anchors per action.

Cells with a true policy-value margin below `.05` are excluded only from
decision-quality aggregates; every cell remains in the evidence.  The seed and
all grids are frozen in code before endpoint generation.

## Comparators

All methods target the probability that the deployed action matches the delayed
latent preference.

1. `raw_immediate`: choose from average immediate-report agreement.
2. `anchor_only`: choose from the delayed anchors alone.
3. `augmented`: estimate each action value as average immediate agreement plus
   the average anchored delayed-minus-immediate residual.

The augmented and anchor-only methods receive the *same delayed anchors*.
Augmented additionally consumes the abundant immediate logs that motivate SDPO.
This is a standard two-phase difference estimator, not a claimed novel theorem.

When copying is zero, the immediate message is a genuine truthful correction and
the augmented learner must exactly match the raw learner.  In the transition
world the immediate report equals the delayed state and should remain useful.

## Frozen gates

The joint developmental signal qualifies only if:

1. coupled raw policies have exactly zero mismatch between worlds;
2. raw expression-world regret on policy-ranking reversals is at least `.03`
   with 16 anchors, establishing a nontrivial problem;
3. with 64 anchors, augmented expression-world reversal regret is at most 60%
   of raw regret;
4. with 16 anchors, augmented expression-world regret is at most 80% of the
   equal-anchor-only regret;
5. with 16 anchors, augmented transition-world regret is at most 50% of
   anchor-only regret; and
6. in zero-copy truthful controls, augmented and raw policies match exactly.

A pass justifies implementing the analogous residual correction inside a
faithful language-model learning setup.  A failure parks this estimator; it does
not refute non-identification.  No threshold may be changed after the run.

