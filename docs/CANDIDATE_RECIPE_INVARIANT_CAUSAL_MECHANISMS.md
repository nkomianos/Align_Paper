# Candidate: recipe-invariant causal mechanisms in model organisms

## Status

**Literature-screen finalist only; no GPU run is authorized.**  This is a
possible successor if the active recency G0 fails or cannot clear the stronger
external-validity standard.  It is not a revival of the recency claim.

## Problem

[The Model Organism Lottery](https://arxiv.org/abs/2607.01033) finds that
mechanistic-interpretability results in language-model model organisms can vary
strongly by training recipe, including when behavioral expression is controlled.
That makes a central open problem operational:

> Can we distinguish a causal feature that is genuinely tied to a behavior from
> a feature that is only an artifact of one way of training that behavior?

The paper establishes the failure mode.  It does not, based on the current
screen, give a held-out method that selects causal mechanisms on one set of
recipes and predicts intervention success on an unseen recipe.

## Hypothesis

For a fixed, benign hidden behavior and matched behavioral expression, an
interventional stability score selected across training recipes will predict
held-out-recipe causal efficacy (signed steering and erasure) substantially
better than a mechanism selected on one recipe by probe accuracy, activation
difference, or within-recipe patching alone.

The target is *cross-recipe causal transfer*, not merely finding a more
interpretable model organism.  A null outcome is useful: it would quantify that
current model-organism causal claims do not transport across their construction
mechanisms.

## Required distinction from existing work

The candidate must show all of the following, otherwise it is closed as a
reanalysis of the Model Organism Lottery:

1. A pre-registered mechanism-selection rule using only training recipes A and
   B, with recipe C completely held out until evaluation.
2. A held-out intervention result: selected components must improve the target
   behavior under the prescribed intervention and beat equal-budget component
   sets selected by the strongest single-recipe baselines.
3. Matched behavioral expression, model scale, data volume, and intervention
   norm across recipes, so a result cannot be explained by an easier organism.
4. A concrete downstream consequence, such as a validity certificate that
   rejects a recipe-specific steering direction or a robust direction that
   transfers to an integrated post-training recipe.
5. Replication on an independent backbone or an independently generated benign
   behavior family.

## Cheap falsifier (J0, not yet authorized)

Use a small public backbone and a harmless two-action, nonce-split task.  Train
the same behavioral target using three materially distinct recipes:

- post-hoc SFT;
- preference or contrastive post-training; and
- integration with unrelated post-training data.

On recipes A/B only, rank candidate residual directions or component sets by a
predeclared cross-recipe stability objective.  Test them once on C using exact
choice likelihood.  Compare against equal-cardinality choices from
single-recipe activation difference, probe, and patching baselines.

The gate passes only if the stability-selected intervention has a signed,
held-out-recipe effect, beats every baseline with uncertainty bounds excluding
zero, and does not damage the ordinary target behavior more than the matched
baselines.  Otherwise the direction is closed without scaling.

## Why it may matter

This would address an evaluation-validity problem that blocks much
mechanistic-safety research.  It has a tight causal test, low initial compute,
and a useful negative result.  Its main risk is that the new selection score is
a superficial statistic rather than a method with real held-out transfer;
therefore J0 must be evaluated before code or a paper claim expands.
