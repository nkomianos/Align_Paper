# External validation plan: causal transport under training-recipe shift

## Status

**Design only; not GPU-authorized.**  This is the paper-scale successor to J0
*if and only if* J0 passes its frozen gate and offline reproduction.  It is not
a way to relabel a failed J0 result.  The goal is to turn the recipe-invariance
idea into a useful empirical contribution rather than a Qwen nonce-routing
demonstration.

## Candidate claim

For a behavior implanted through several training recipes on a common
backbone, agreement of independently derived, **base-subtracted** residual
updates predicts which intervention will causally alter the behavior in a
previously unseen training recipe better than equally budgeted single-recipe
directions.

The claim is deliberately about transport of an *intervention-selection rule*,
not identity of a universal circuit.  It asks a different question from
forecasting a mechanism along one SFT trajectory and from measuring whether a
model organism is easy to interpret.

## External testbed and frozen source

The initial testbed would be the released 54-model suite accompanying
[The Model Organism Lottery](https://arxiv.org/abs/2607.01033), with source
repository commit `9384b9231580f43c77a5f9bf7a7339750b15ab5c`.  It provides
matched model-organism variants across integrated DPO, post-hoc DPO,
demonstration fine-tuning, and synthetic-document fine-tuning; it also includes
multiple behavior families, OLMo2-1B and Gemma-3-1B backbones, and CakeBake
training-order replicates.  The experiment must pin every Hugging Face model
revision in its registry before inference starts.

This is an external validation set, not a source of tunable examples.  We may
use each organism's documented trigger prompts to measure a known behavioral
effect, but no held-out recipe's behavior score, response examples, or hidden
states may influence direction/layer selection.

## Per-fold protocol

For each available `(quirk family, backbone, training-order seed)` group with
at least three construction recipes:

1. Select an ordered A/B source pair and a disjoint held-out recipe C.  Freeze
   all C artifacts before source processing.
2. On a fixed neutral prompt pool and matched trigger/control prompts, derive
   each source's per-layer update relative to its adapter-disabled/base model.
   Remove the shared prompt component before comparing sources.  Raw
   `TARGET_A - TARGET_B`-style prompt directions are prohibited.
3. Choose exactly one layer and normalized direction from A/B agreement.  The
   selection score and all candidate scores are written before C is loaded.
4. Apply signed steering and projection erasure to C on unseen trigger and
   matched non-trigger prompts.  The primary outcome is the paired change in
   the suite's behavior-expression score; secondary outcomes are causal
   necessity, unrelated-behavior preservation, and blinded investigator
   recoverability.
5. Compare against frozen, equal-norm controls: A-only, B-only, random,
   principal component, label-shuffled, and a C-derived **oracle ceiling** that
   is report-only and never eligible for selection.

Every fold is independent: no direction, layer, scale, prompt filtering rule,
or threshold is chosen after inspecting its C outcome.  A fold that lacks a
valid behavior-expression control or a pinned source checkpoint is excluded
before outcomes are opened, with the reason logged.

## Primary analysis and decision rule

The unit of inference is the group-fold, not individual prompt generations.
For each fold, estimate the signed C steering effect and paired erasure effect
with a within-fold bootstrap.  Aggregate only after all folds are fixed using a
hierarchical bootstrap over groups then prompts.

A paper-scale green light requires all of the following:

- the A/B rule beats the best source-only direction in the prespecified
  aggregate C steering effect, with a confidence interval excluding zero;
- the advantage is present on both backbone families and at least two distinct
  quirk families, rather than driven by CakeBake or a single recipe pair;
- erasure supplies a corresponding necessity signal and neither intervention
  produces unacceptable unrelated-behavior loss;
- results survive the published training-order replicates; and
- the comparison against a probing/future-localization-style baseline described
  in the novelty audit is technically feasible and not superior.

Failure of any item is a no-go for an ICLR method claim.  A null result may be
worth a short empirical note only if it is broad, precisely estimated, and
explains why common model-organism evaluation cannot validate transport.

## Compute and access prerequisites

The released suite contains roughly 1B-parameter checkpoints.  A pilot should
use one quirk group and three recipes, then a full study loads models
sequentially and caches activation summaries rather than keeping all checkpoints
resident.  A GH200 or H100 has ample memory for the pilot; the full matrix is
primarily an inference/storage task, not 9B fine-tuning.  Before authorizing it
we need:

- a Hugging Face token for reliable, rate-limit-free retrieval of the public
  checkpoints (read-only access is sufficient);
- at least 200 GB of free local/attached disk for pinned checkpoints, source
  data, and immutable activation/evidence artifacts; and
- a fresh isolated checkout and output root so the active G0 evidence remains
  untouched.

This plan deliberately does not request or download those resources while G0
is running.
