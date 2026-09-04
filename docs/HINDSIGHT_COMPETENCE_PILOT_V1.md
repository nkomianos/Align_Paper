# Capability-controlled learning pilot — freeze before running

This follows the failed practical estimator-split diagnostic, not a new paper
green light. The previous small model was only 4/8 correct on its training tasks;
improvement from both feedback objectives may have reflected ordinary task
learning. Test preservation of acquired competence before attributing benefits
or damage to the feedback objective. No claim of novelty for competence effects:
[SC-SDPO](https://arxiv.org/abs/2605.27765) already studies difficulty-aware
self-distillation. [Privileged Likelihood](https://arxiv.org/abs/2608.09263)
already separates feedback dependence, score meaning, and training behavior.

## Frozen apparatus

Public cached Qwen3-0.6B at the previously pinned revision, CPU float32/eager,
rank-4 alpha-8 attention LoRA, seed 9046401. Fresh procedural numeric choices:
price, duration, capacity, distance; explicit higher/lower preference and paired
option orders. Distinct numeric pairs within each domain across all splits.
64 warmup examples, 32 qualification, 32 feedback adaptation, 64 final evaluation.
Final evaluation uses a different fixed wording. This is synthetic transfer
within four simple task families, not an external benchmark or human welfare.

Warmup: two epochs, batch eight, AdamW 1e-4, zero weight decay, gradient clip one.
Evaluate qualification only after the fixed 16 updates. Require >=90% aggregate,
>=75% in each domain, and >=.95 A/B vocabulary mass for every qualification
example. If not met, preserve evidence and stop before any feedback training.
No reranking warmup checkpoints using final evaluation.

If competent, save the model and freeze all feedback-teacher distributions from
that checkpoint. Use the released hindsight wrapper and a known simulator law
mixing 90%-truthful feedback with action copying: strengths zero and .9.
The user's preference does not transition. This isolates expression effects.
Exact population gradients integrate the known simulator channel, avoiding
Monte Carlo variance. This is an oracle-expectation controlled experiment, not
a deployable learner that estimates its feedback channel from logs.

Compare own-response scoring, full reverse KL, full KL with a standard anchor
gradient half-space projection, and anchor-only supervised learning. Each starts
from the same competent checkpoint, same eight batches, eight SGD ascent updates,
learning rate .01 and gradient norm clip one. No per-arm tuning. Teacher frozen
throughout; no adaptive teacher, full-vocabulary SDPO, or free-text generations.
Two of eight examples in each batch expose ground-truth preference anchors to
the two anchor methods. Same examples/labels and two passes; report unique as
well as exposure counts. Simulator labels are used to integrate the channel,
not supplied to raw learners as target actions. No-adaptation is also reported.

Projection onto g dot g_anchor >=0 is a standard first-order retention baseline,
not our new algorithm; see [GEM](https://arxiv.org/abs/1706.08840). It guarantees
only local nonnegative alignment on that batch, not finite-step generalization
or safety. Anchor-only is label-budget matched, not compute matched; it is cheaper.

## Interpretation rules

Report every arm's heldout accuracy, original-request NLL, domain results and
vocabulary mass. Inspect loss relative to the competent checkpoint and relative
to its zero-copy control. Do not count a drop common to both channels as evidence
that action dependence caused it. Report projection against anchor-only: a fix
that merely beats an unanchored learner does not establish useful new learning.
Keep all outcomes even if the direction is inconvenient. One seed and eight
updates cannot establish a stable phenomenon, acceptance probability or a theorem.
No automatic GPU expansion, regardless of the result. Save adapters, warm optimizer,
all step probabilities/coefficients, prompts, source, runtime and manifest.

## Separate idea rejected before compute

Ambiguous tool failures and unsafe retries are directly covered by
[Verified Tool Calls](https://arxiv.org/abs/2608.02645),
[Resume Means Resume](https://arxiv.org/abs/2608.03836), and the public
[IdempotencyBench](https://github.com/gssanjana4/idempotencybench).
Do not add another timeout/idempotency benchmark to the GPU queue as a novel
candidate without a materially different contribution. No experiment was run
for that rejected idea; these are primary-source collision findings.

## Completed and verified — scope limit discovered

Commit `3d32159`; root `artifacts/hindsight_competence_cpu_v1`. Completed
1,250 forwards, 544 backwards, 16 warmup and 64 feedback-stage updates in
1022.4 seconds (~17 minutes), all on the laptop CPU. All adapters, warm optimizer,
data, prompts, probabilities and logs are retained. Read-only verifier passes
checksums, coefficients, outcomes, batch/anchor accounting, checkpoint validity
and the exact independent-feedback null (own/full final adapters identical).
It does not independently rerun model inference or optimizer steps.
Receipt `artifacts/hindsight_competence_cpu_v1_verified.json`; manifest SHA
`1e434e319d625cf869892664da6776e64d70bff20cbf0e57711453c553184675`.

The qualification check passes: 30/32, versus 16/32 before warmup. However,
the no-adaptation checkpoint scores only **36/64 = 56.25%** on the separate,
differently rendered final evaluation (NLL 2.59856). Numeric values and rendering
both differ from qualification; this is not a paired proof that wording alone
caused the drop. It shows that competence did not transfer sufficiently to the
intended evaluation. Do not describe this as preservation of a robustly acquired
general ability. The frozen qualification decision remains unchanged; the final
scientific interpretation is weaker than that engineering prerequisite.

| Method | Independent-feedback accuracy | Copying .9 accuracy | Independent NLL | Copying .9 NLL |
|---|---:|---:|---:|---:|
| Own-response | 39/64 (60.94%) | 36/64 (56.25%) | 2.28610 | 2.16949 |
| Full reverse KL | 39/64 (60.94%) | 40/64 (62.50%) | 2.28610 | 1.87298 |
| Anchor-projected full KL | 40/64 (62.50%) | 40/64 (62.50%) | 2.24728 | 1.90514 |
| Anchor-only | 36/64 (56.25%) | 36/64 (56.25%) | 2.88567 | 2.88567 |

Anchor methods use 16 label exposures covering 15 unique contexts out of 32
adaptation contexts; the paired option orderings are not independent tasks.
No claim of 25% unique-label coverage is warranted. All teacher A/B mass is
at least .999989; the issue is not unrestricted-vocabulary format collapse.

Own-response loses three correct decisions versus its independent-feedback
control but improves NLL; full KL improves both metrics. Projection does not
beat unprojected full KL in the copying arm, and is slightly worse in NLL.
Do not select accuracy alone to manufacture a uniform harmful-feedback story.
This is neither the report's required benefit-plus-harm phenomenon nor a
successful corrective method. No human preference transition is represented.

PI decision: no paper or expensive training expansion. The experiment now has
actual learned parameter updates, but tiny single-model results and weak transfer
do not justify causal-welfare or high-confidence acceptance claims. A future
learning study needs competence validated on the intended evaluation distribution
before feedback updates, stronger models and a useful contribution beyond known
gradient projection/feedback-dependence observations. Preserve this entire run;
do not retune its checkpoint or replace its evaluation after seeing the outcome.
