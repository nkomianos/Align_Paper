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
