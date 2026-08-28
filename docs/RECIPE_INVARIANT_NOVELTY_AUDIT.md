# Novelty and identification audit: recipe-invariant mechanism selection

## Current decision

**Retain only as a conditional, high-bar finalist—not a paper claim.**  The
direct literature screen supports a potentially distinct but now substantially
more demanding contribution: select a causal intervention on recipes A/B and
prospectively predict its efficacy on an unseen, matched training recipe C.
J0 is an operational check, not enough evidence to justify a paper-scale run.

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

[Beyond Static Interpretability](https://arxiv.org/abs/2608.24482), posted on
25 August 2026, is an additional close risk.  It forecasts post-SFT
task-critical parameters from a base-model/dataset pair, then uses the forecast
to guide tuning.  It is not a recipe-shift experiment: its target is a future
state along a single target-SFT trajectory, and it does not select on completed
recipes A/B before evaluating causal intervention in a held-out construction
recipe C.  Nevertheless, it means that a bare claim of “prospective mechanism
prediction” is no longer differentiating.

[Mechanistic Data Attribution](https://arxiv.org/abs/2601.21996) traces
interpretable units to influential *training samples* and alters their
emergence by removing or augmenting those samples.  It is a close conceptual
neighbour because it addresses causal training origins, but it neither holds a
post-training objective out from direction selection nor tests whether a
source-recipe intervention transfers to that unseen construction.  A
recipe-invariance paper must therefore compare against source-only directions,
not claim credit for the broader idea that training affects mechanisms.

[Certified Interventional Fidelity](https://arxiv.org/abs/2607.08349) is a
methodological rather than topical collision: it formalizes causal estimands
and confidence sequences when intervention evaluations are monitored or
adapted.  J0 uses a fixed prompt set, two frozen seeds, a fixed intervention
scale, and one terminal gate, so its paired bootstrap is appropriate for the
initial diagnostic.  Any expanded study that adaptively samples or stops by
observed effects must adopt an anytime-valid analysis or preserve the fixed
terminal design.

Consequently, any paper-scale version must establish all of the following,
before claiming a contribution:

1. **Recipe OOD, not time extrapolation:** the selection rule is frozen without
   access to recipe C, and every recipe-C causal metric is held out from
   selection and tuning.
2. **A substantive comparison:** it includes a faithful future-localization or
   probing-SFT-style predictor where feasible, in addition to random,
   PCA, and single-recipe controls.  Merely beating weak direction controls is
   insufficient.
3. **External validity:** it repeats on a released model-organism family or a
   comparably realistic integrated construction, across a second model family
   and independently generated behavior family.  A single nonce-routing toy
   result is only a diagnostic.
4. **Claim discipline:** it claims predictive validity of an intervention
   selection procedure under a defined training-recipe shift, not a universal
   mechanism or a method to forecast arbitrary future circuits.

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
seeds, this direction is closed.  A positive J0 result licenses only a fully
offline reproduction and the stronger comparison/external-validity program
above.  It cannot by itself support an ICLR claim.
