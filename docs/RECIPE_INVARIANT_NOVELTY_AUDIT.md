# Novelty and identification audit: recipe-invariant mechanism selection

## Current decision

**Retain as a conditional finalist, not a paper claim.**  The direct literature
screen supports a narrow, falsifiable contribution: selecting a causal
intervention on recipes A/B and predicting its efficacy on an unseen, matched
training recipe C.  J0 must pass before this direction receives any additional
compute.

## Closest work and distinction

[The Model Organism Lottery](https://arxiv.org/abs/2607.01033) trains 54 model
organisms using post-hoc SFT, DPO, and integrated DPO and shows that
interpretability depends on objective, behavior, architecture, and data
generation.  That is the motivating failure mode.  Its reported task is to
benchmark interpretability methods across organisms—not to predeclare a
selection rule on two recipes and test whether that rule identifies an
intervention which transfers to a third, held-out recipe.

[Pattern Selectivity is Not Task-Causal Structure](https://arxiv.org/abs/2606.05378)
evaluates a screen-and-ablate recipe across three model families and finds that
the screening procedure can port while the primary causal circuit does not.
It is a crucial negative comparator.  It does not use matched post-training
recipes for one backbone/behavior, nor hold one recipe out from component
selection.  J0 therefore cannot claim universal mechanisms or cross-model
component identity; it asks only whether a selection *procedure* improves
held-out causal intervention over equal-budget single-recipe procedures.

## Identification discipline

[Lin and Liu](https://arxiv.org/abs/2605.08012) argue that validation metrics
are not, by themselves, causal identification.  Accordingly J0 makes no claim
that a stable activation direction is "the" implementation of a behavior.  Its
causal conclusion is limited to its concrete interventions: signed steering,
projection erasure, matched-direction controls, preserved unrelated behavior,
and transfer to the held-out construction recipe.  The inference relies on the
synthetic-task and intervention-family assumptions; it cannot establish causal
transport to natural harmful behaviors or arbitrary models.

## Kill condition

If recipe-C effects do not beat all equal-budget baselines for both frozen
seeds, this direction is closed.  A positive result still requires a fully
offline reproduction, an independent model family, and a separately generated
behavior family before it could plausibly support an ICLR claim.
