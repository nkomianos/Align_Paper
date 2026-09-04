# Neural delayed-anchor SDPO G0 protocol

## Purpose

This is the next mechanism test after the positive finite-state delayed-anchor
DEV. It asks whether the same two-phase correction works on the paper's
full-distribution self-distillation objective, rather than on tabular agreement
means.

The implementation follows the public SDPO repository at commit
`3b17d2a67bd2565b9fbda495fd16a485406aa954`: the student distribution is
`pi(.|x)`, the detached hindsight teacher is `pi(.|x,o)`, and each token uses
reverse `KL(student || teacher)`. The G0 uses the first answer token, full
vocabulary, and the repository's exact hindsight block. It is a faithful
single-decision-token instance of the mathematical objective, not a free-form
reproduction of the paper. The released online default approximates this KL on
the student's top 20 tokens plus a tail bucket. This gate deliberately uses the
exact full vocabulary to remove that approximation as a failure mode; a pass
still requires replication with the released approximation and full sequences.

## Frozen data-generating process

The anonymous population prefers the detailed semantic option initially with
probability `.75`. Logged concise/detailed actions are balanced within each
initial state. When a concise action conflicts with the initial preference, the
immediate report always copies it (`c0=1`); a conflicting detailed action is
reported truthfully (`c1=0`). Therefore:

- expression world: delayed preference remains initial; detailed is optimal;
- transition world: copied reports persist; concise is optimal;
- immediate `(prompt, action, report)` logs are exactly identical.

There are 128 training interactions across 16 natural-language task surfaces.
Thirty-two delayed anchors are selected by within-stratum hashes, retaining the
population proportions exactly. Thirty-two evaluation prompts cross all surfaces
with both A/B orderings. Feedback names the semantic response style, not its
letter, so success cannot come from a fixed A/B bias.

The fixed 64-step schedule has batches of 16 with exactly four anchors. Every
anchor and non-anchor appears eight times. Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` receives rank-8 LoRA adapters on both
full-attention and DeltaNet projections; learning rate is `1e-4`, with identical
initial adapters and fresh AdamW state per arm.

## Losses and equal-anchor comparisons

For immediate hindsight loss `L_O` and delayed-anchor loss `L_B`:

- raw SDPO: `mean_all L_O`;
- anchor-only SDPO: `mean_anchor L_B`;
- anchor SFT: supervised delayed semantic labels on the same anchors;
- augmented SDPO: `mean_all L_O + mean_anchor(L_B-L_O)`.

The last expression is an unbiased two-phase estimate of the anchor-defined
population loss under random auditing. Anchor-only, anchor SFT and augmented
SDPO see the identical 32 delayed labels and identical repeated exposure budget.
Raw SDPO is shared by the two observationally equivalent worlds. In the
transition world `B=O`, so augmented loss must equal raw loss exactly. A separate
zero-copy truthful-feedback arm is the acquisition positive control.

## Qualification and decision boundary

Before training, the frozen model must map semantic feedback to the correct
swapped A/B option on every evaluation prompt, with minimum normalized target
probability `.70`, mean at least `.90`, and minimum full-vocabulary A/B mass
`.20`. Failure stops before updates.

The method signal requires all of the following on evaluation wording:

1. truthful raw SDPO reaches at least 80% semantic detailed accuracy and improves
   detailed normalized probability over the base model by at least `.10`;
2. the shared raw arm reaches at least 80% concise accuracy, demonstrating the
   predicted immediate-feedback policy (right for transition, wrong for
   expression);
3. expression augmented SDPO reaches at least 80% detailed accuracy;
4. its detailed probability exceeds the better of equal-anchor SDPO and anchor
   SFT by at least `.05`, without worse accuracy; and
5. the transition augmented/raw loss identity passes at every audited batch.

A failure with a qualified teacher and positive control parks this correction.
A pass authorizes multiple seeds, the released top-20-plus-tail loss, and a
free-form/user-renderer replication; it does not qualify the paper or establish
human prevalence. No threshold may be changed after endpoint generation.

The runner checkpoints every completed arm and writes partial step logs plus the
current adapter and optimizer on an exception. Failed evidence is therefore
preserved for diagnosis, but never silently promoted into a gate decision.
