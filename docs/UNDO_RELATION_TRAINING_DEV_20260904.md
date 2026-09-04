# Correct local-relation training: bounded developmental comparison

This supersedes the operational parking recommendation in UNDO_PI_REAUDIT,
not its scientific cautions. The user explicitly authorized experiments within
the remaining GPU window. No model result is present at protocol creation.
No automatic paper green light or expansion follows from this run.

## Hypothesis and controls

Can local, semantically valid rewrite-distribution matching learned on 2-5
updates improve final-state decisions after 20-100 updates more than matched
terminal-answer SFT or full canonical-state teacher matching? Unlike the
original proposed cancellation identity, the implemented law is sound for all
initial states: two consecutive updates to the same field reduce to the last
update. This covers set/set and set/clear, not rollback to an older value.

All three arms have identical student histories, labels, training-example
order and optimizer-update budget. The local teacher sees only the first of
two consecutive writes removed; unrelated updates remain. The canonical
teacher sees the entire current state. Teacher distributions are frozen before
adapter optimization. SFT targets one answer token; KL uses the full vocabulary
at that one next-token position. Therefore the canonical arm is a **single-action
canonical distillation proxy**, not a sequence-level/on-policy reproduction of CCOPD. Report teacher preprocessing
time as well as training and evaluation cost; equal student tokens do not imply
equal total compute.

## Fixed data

Generator: `src/interaction_sprint/undo_relation_data.py`, seed9041741.
Prepared root: `artifacts/undo_relation_training_v1`.

- 512 training histories, 2-5 edits, four surfaces and two relations.
- 40 separate development cases (eight matched quintuplets, depth20).
- 320 final cases: 64 matched quintuplets, depths4/20/60/100; fresh field IDs,
  random noise values and a different command renderer from training.
- Five conditions: history, canonical state, matched-operation-count padding,
  explicit final update, one-update counterfactual.

Final data are generated and hashed before optimization. Do not remove cases,
change rendering, choose checkpoints, or tune hyperparameters using final
results. All splits are synthetic post-pilot DEVELOPMENT, not a pristine
confirmatory test or human-dialogue external validation. Surface names alone
do not establish four fundamentally different tasks; all are register tracking.
Padding matches update count, not token count. The exact canonical/state
tracking baseline already solves formal operations; it is not a novel method.

The training module is a separate implementation. Root must freeze its exact
optimizer, seed, train steps, batch size, learning rate, LoRA configuration and
model revision before starting, and archive that configuration. Use the same
pinned Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`
for continuity if its environment is available. One seed tests engineering
feasibility, not reproducibility. Do not silently substitute model families.

## Evaluation and decisions

Record the four answer probabilities, full-vocabulary mass on those letters,
predicted letter, and ground-truth-free input IDs for every checkpoint/case.
No free-text parser is needed. Report mean choice mass and canonical, padding,
and counterfactual accuracy separately, rather than allowing a restricted
softmax to hide refusal/format failure.

Development qualification: if baseline controls are below90% or mean answer
mass below0.5, label the setup unqualified and inspect rather than declaring
the hypothesis false. Do not replace or drop hard cases to qualify it.
Final reporting keeps all cases regardless of control outcome.

Primary descriptive comparison is local versus both trained baselines on
depth60/100 history accuracy, with paired per-history differences and bootstrap
intervals. Also report depth4/20, stale-choice rates, and all controls. A local
gain accompanied by counterfactual or canonical regressions is not a clean
success. No arbitrary threshold turns one additional correct example into
a fundamentally different scientific conclusion. Report counts and intervals.

A positive developmental result licenses only three-seed replication and a
full-vocabulary, data/compute-matched strong distillation comparator plus
external natural-language update tasks. A null result says this local law,
training budget and one-action apparatus did not outperform baselines; it does
not prove the entire idea impossible. No uniform length-generalization theorem
is claimed: short-prefix training does not establish reachable-state coverage.

Reserve time within the user's eight-hour window for checkpoint/evidence
retrieval and hash verification. Never delete incomplete artifacts. Runtime
estimates must come from the actual training throughput probe, not invented
H100-hour numbers.
