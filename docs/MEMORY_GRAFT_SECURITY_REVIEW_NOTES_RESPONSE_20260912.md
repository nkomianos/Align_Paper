# Response ledger for optimizer-section review notes

## 1. Quality cost at the aggressive table rate

**Accepted.** Existing post-training outcomes already show a heterogeneous cost.
At 410M, four seeds retain clean NLL 2.90--2.93 and 100% matched-benign
accuracy, while one seed reaches NLL 3.49 and 82.4% accuracy. G6 now provides
the missing paired measurement. At the strongest rate, post-minus-pre clean NLL
rises 0.0773 [0.0514, 0.1150] at 410M and 0.06894 [0.06691, 0.07096] at 1.4B.
All four rates and intervals are reported in the paper.

## 2. Five-seed endpoint uncertainty

**Completed, with a statistical correction.** G6 ran 16 new seeds per size.
The endpoint prevalence estimate pools them with five already verified seeds
for n=21 and worst-case Wilson half-width 0.1967. The new panel remains
separately visible. G6 does not use a Student-t lower bound to summarize the
near-binary route indicators.

## 3. Bimodal seed-level routes

**Completed in substance; mechanism wording narrowed.** The revised analysis
reports every seed and the prevalence of whole-table, final-row,
earlier-row, history-row, and shared-prefix-control effects above the registered
0.15 meaningful-effect scale. This supports seed-variable route selection under
a fixed recipe. It does not by itself prove a discrete latent mechanism or
unpredictability outside the sampled seeds.

## 4. G5 prominence

**Accepted.** G5 now has its own results subsection, and the abstract states the
two-layer 5x-table result directly.

## 5. Learning-rate sweep visibility

**Completed.** G6 repeats baseline, 1e-3, 1e-2, and 1e-1 under paired poison
data and dropout RNG for 16 seeds per size. The transition is sharp and
scale-dependent: 15/16 410M seeds are table-dependent by 1e-2, compared with
2/16 at 1.4B. Every endpoint is reported.

## 6. Title and abstract

**Accepted.** The title is now *Addressing Is Not a Security Boundary:
Optimizer Policy Moves the Storage Component, Not the Deletion Boundary*. The abstract is
reduced to three findings: ordinary backbone routing, a functional deletion
positive control, and optimizer-dependent component movement without a
reliable item boundary.

## Smaller notes

The main text now identifies the invalid 1.4B developmental selection as
evidence of a numerically sharp routing regime. Related work now motivates the
experiment as a necessary condition for deletion, audit, and isolation
guarantees instead of debating whether prior papers explicitly promised the
converse.

## Second review round — 13 September 2026

1. **Hase et al. added.** Related work now cites *Does Localization Inform
   Editing?* and distinguishes its inferred dense-model localization from our
   exact, pre-forward hash address. This sharpens the novelty boundary: even a
   correct architectural read address need not be the learned write location.
2. **Quality cost moved into the abstract and normalized.** The strongest-rate
   NLL changes are 2.66% and 2.64% of the respective mean pre-poison NLL and
   imply 8.04% and 7.14% geometric-mean perplexity increases.
3. **Knife-edges disclosed.** The pooled 410M whole-table sufficiency and
   410M history-sufficiency calls are both 15/21 with Wilson lower bound
   0.500436; one seed flip reverses either call. The manuscript labels both
   borderline and states that a majority-prevalence result is not a security
   guarantee.
4. **Pooling provenance made explicit.** G6's primary evidence is the 16 new
   seeds. The 410M pool adds only the valid G3 AdamW `1e-1` seeds, and the 1.4B
   pool adds only the separately fixed G3.1 seeds. The original invalid G3 1.4B
   selection is excluded. Poison training and evaluation match, and the 1.4B
   clean recipe is identical. The audit found that G3/G4's 410M checkpoints use
   10M clean-adaptation tokens versus G6's 5M; the paper now discloses this and
   treats the 410M/G4 pools only as heterogeneous secondary consistency checks.
5. **Direct-write quality measured.** A post-hoc same-checkpoint audit finds
   paired clean-NLL changes of -1.00e-5 and +1.21e-5 nats at 410M and 1.4B,
   with perplexity ratios 0.999990 and 1.000012. The paper labels this audit
   post hoc and uses it as a constructive surgical-write result.
6. **Verifier promoted to a contribution.** The symmetric component-swap and
   controlled row-ablation protocol is now named as a post-adaptation
   storage-location verifier and presented as an operator-facing deliverable.
7. **Presentation fixes completed.** The temporal footprint has its own
   subsection. The seed table now includes whole-table, outside-table, and
   final-row sufficiency alongside necessity for every new seed. The 1.4B seed
   03 value of 0.149414 is daggered and explained. The prior merged-column
   formatting is gone.
8. **Multiplicity stated.** No multiplicity correction is applied because the
   estimands are separately registered and all are reported; marginal calls are
   explicitly identified and are not generalized across estimands.
