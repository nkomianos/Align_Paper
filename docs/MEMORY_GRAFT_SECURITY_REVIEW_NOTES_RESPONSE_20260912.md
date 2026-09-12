# Response ledger for optimizer-section review notes

## 1. Quality cost at the aggressive table rate

**Accepted.** Existing post-training outcomes already show a heterogeneous cost.
At 410M, four seeds retain clean NLL 2.90--2.93 and 100% matched-benign
accuracy, while one seed reaches NLL 3.49 and 82.4% accuracy. At 1.4B, mean NLL
is 2.6816 and mean benign accuracy is 99.90%. These absolute outcomes will be
reported. G6 adds the missing within-seed pre-poison checkpoint measurement at
every learning rate, so the paper can report paired quality deltas rather than
make an unreliable cross-run comparison.

## 2. Five-seed endpoint uncertainty

**Accepted, with a statistical correction.** G6 fixes 16 new seeds per size.
The endpoint prevalence estimate pools them with five already verified seeds
for n=21 and worst-case Wilson half-width 0.1967. The new panel remains
separately visible. G6 does not use a Student-t lower bound to summarize the
near-binary route indicators.

## 3. Bimodal seed-level routes

**Accepted in substance; mechanism wording narrowed.** The revised analysis
will report every seed and the prevalence of whole-table, final-row,
earlier-row, history-row, and shared-prefix-control effects above the registered
0.15 meaningful-effect scale. This supports seed-variable route selection under
a fixed recipe. It does not by itself prove a discrete latent mechanism or
unpredictability outside the sampled seeds.

## 4. G5 prominence

**Accepted.** G5 now has its own results subsection, and the abstract states the
two-layer 5x-table result directly.

## 5. Learning-rate sweep visibility

**Accepted.** The parent one-seed screen suggests an abrupt transition, but it
cannot establish a dose response. G6 repeats baseline, 1e-3, 1e-2, and 1e-1
under paired poison data and dropout RNG for 16 seeds per size. Every endpoint
is registered for publication.

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
