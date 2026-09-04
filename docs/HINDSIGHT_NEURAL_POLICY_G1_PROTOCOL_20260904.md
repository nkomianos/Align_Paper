# Hindsight neural policy-learning G1 protocol

## Role in the decision ladder

This v2 protocol supersedes the unrun v1 at commit `e00a159` before any G1
endpoint exists. A prospective code audit found that v1 repeatedly inserted all
eight anchors into a nominal 16-record population batch. Its implemented loss
would therefore have overweighted the fixed panel and would not have been the
claimed population expectation plus paired correction. V2 separates those two
samples exactly; v1 must never be run or interpreted.

This protocol may run only after the
nested-budget neural-gradient G0 v2 returns `NEURAL_GRADIENT_G0_V2_QUALIFIED` and
its complete evidence passes the committed read-only verifier. G0 asks whether
the sparse paired estimator points in the right neural-gradient direction; G1
asks the more important question: does repeatedly applying that estimator move
an actual policy toward the full delayed-feedback oracle?

G1 is not automatically authorized by this document. A launcher must receive a
qualified, checksum-valid G0 root. Interface failure or a scientific G0 failure
stops before policy training.

## Frozen design

The model, revision, rank-8 LoRA targets, exact first-answer-token full-vocabulary
reverse-KL objective, prompt, and semantic-interface qualification are unchanged
from G0 v2. The 128 training interactions and 32 swap-paired evaluation prompts
are also unchanged.

Eight disjoint panels contain eight delayed anchors each, four per randomized
logged action. Panel membership is a hash of row identity and is blind to latent
state and all outcomes. For each panel, all three sparse learners receive exactly
the same eight delayed labels:

1. anchor-only SDPO;
2. anchor-only supervised first-token learning; and
3. augmented SDPO, using the full immediate-feedback term plus the paired
   delayed-minus-immediate correction on those anchors.

Raw immediate SDPO and full delayed-oracle SDPO are global controls. Every arm
starts from the identical zero-B adapter and runs 32 AdamW updates at `1e-4`.
The fixed population schedule contains 16-record batches balanced on logged
action; every record appears exactly four times. Each augmented update computes
the raw immediate term on one such population batch and independently computes
the delayed-minus-immediate correction on all eight anchors in its fixed panel:

`mean_population L_immediate + mean_panel(L_delayed - L_immediate)`.

The repeatedly measured anchors are never substituted for members of the
population batch. Anchor-only learners use only the same eight panel anchors.
Thus all three sparse learners share exactly the same delayed labels, while only
the augmented learner also uses the identical population log available to raw
SDPO.

There are 26 trained arms: two global controls and three learners for each of
eight panels. Adapters, optimizer states, step logs, evaluation rows, failure
state, sources, and checksums must all be preserved.

## Prospectively frozen decision

Semantic action 1 is the persistent-preference oracle target; raw immediate
feedback favors semantic action 0. All probabilities are normalized within the
native A/B answer tokens after undoing option order.

G1 qualifies only if every condition holds:

1. the full delayed oracle raises mean action-1 probability by at least `.10`
   from the common baseline;
2. raw immediate SDPO lowers it by at least `.10`;
3. the oracle-minus-raw endpoint gap is at least `.25`;
4. augmented SDPO beats the better of equal-anchor SDPO and SFT in at least six
   of eight panels;
5. its median action-1 advantage over that best comparator is at least `.05`;
6. median absolute distance to the oracle endpoint falls by at least 20%;
7. mean absolute distance to the oracle endpoint falls by at least 20%;
8. every endpoint retains A/B probability mass of at least `.10` and at least
   half the baseline minimum mass; and
9. the maximum option-position gap is no more than `.10` or baseline plus `.02`,
   whichever is larger.

Thresholds and schedules cannot change after any G1 model endpoint is produced.

## Interpretation and next decision

A pass would establish a controlled synthetic neural policy-learning effect:
with the same sparse delayed labels, the paired correction outperforms both an
SDPO-only and a supervised anchor baseline and approaches the full oracle. It
would move Hindsight from conditional yellow to a serious paper candidate, but
would not by itself justify submission. The next required evidence would be a
released top-20-plus-tail or full-sequence implementation, multiple random seeds,
a second model family, and the separately bounded longitudinal-human analysis.

A qualified interface and G0 followed by G1 failure parks the current correction
instead of changing thresholds. Failure of raw/oracle acquisition invalidates
the policy assay on this interface rather than disproving the causal theorem.

Expected GH200 time is approximately 4--8 hours after model/environment setup.
This is deliberately more expensive than G0 because it measures 832 actual
optimizer updates instead of only initial gradients.
