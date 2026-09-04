# Delayed-anchor selection and contamination DEV protocol

## Question

The positive delayed-anchor experiment assumes that delayed audits are sampled
at random and measure the intended persistent preference perfectly. This frozen
CPU audit separates two violations:

1. **Post-report selection:** whether an anchor is observed depends on the
   immediate report. Under positive known propensities and
   `S independent of B conditional on A,O`, inverse-propensity weighting should
   identify the observed delayed-anchor target.
2. **Anchor contamination:** the delayed measurement itself partially copies
   the immediate report. Weighting cannot recover the uncontaminated target;
   sufficiently strong contamination can reverse its policy ranking.

For action `a`, immediate agreement `O`, delayed agreement `B`, observation `S`
and known propensity `e(a,O)`, the corrected estimator is

`mean(O) + mean[S/e(a,O) * (B-O)]`.

It is compared with raw immediate feedback, the unweighted residual estimator,
unweighted anchors, and Horvitz--Thompson anchor-only estimation. Every method
uses the same realized anchors. The experiment evaluates initial-preference
utility and separately reports error against the potentially contaminated
anchor target.

## Frozen grid and criteria

The grid has four initial preference rates, two copying rates per action, five
contamination levels and four selection-bias levels. Each cell contains 1,000
coupled repetitions with 512 immediate observations per randomized action and
12.5% base anchor probability. At the largest selection bias, anchor
propensities are `.03125` after disagreement and `.21875` after agreement.

The characterization qualifies only if all six hold:

- with clean random anchors, IPW augmentation lowers regret by at least 10%
  versus equal-anchor IPW;
- post-report selection creates at least `.02` bias in the naive augmented
  estimate somewhere on the frozen clean grid;
- known-propensity augmentation has maximum clean-grid bias at most `.012`;
- at maximum selection bias its mean bias is at most 35% of naive augmentation;
- rank-preserving contaminated targets retain at least 80% correct policies;
- after a material anchor-target rank flip, at most 20% of policies still match
  the uncontaminated target.

The final criterion is a limitation check, not a desired safety failure: a
successful estimator of a corrupted target should reveal that identification
of the wrong estimand is not robustness.

## Decision boundary

A pass characterizes where random invitations or known response propensities can
repair selection and where they cannot repair measurement contamination. It is
not neural evidence, a claim about human prevalence, an algorithmic-novelty
claim, or a paper green light. Thresholds and grids are frozen before the first
endpoint run.

