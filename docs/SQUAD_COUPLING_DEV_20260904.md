# Native cross-tokenizer coupling: stochastic free-answer DEV

This is the first actual pretrained-model stochastic comparison for the candidate
in [the theory/novelty protocol](BYTE_CLOCK_COUPLING_DEV_20260904.md), not a paper
acceptance gate or a claim to improve model accuracy. No paid GPU is used.

## Why this task rather than the previous qualification questions

The earlier native byte-decoder qualification completed16 outputs/124 forwards
in162.04s, all terminating, no silent tokens. Frozen source/model/evidence checks
and sampling replay from saved logits passed; manifest
04753f1dc8692d4733d8a8e353bff02201912324fb756cb4fd3b7efa2dd750e3.
Exact-option-only format compliance was1/8 Qwen and0/8 SmolLM; raw answers were
mostly letter-prefixed. Preserve those strict scores, but do not interpret them
as near-zero comprehension. Most decisions concerned a common single letter,
so the format is poorly suited to testing multi-token cross-vocabulary coupling.

The new public SQuAD1.1 DEV asks free-answer reading comprehension with a supplied
passage. It uses the standard normalization, exact match and word-overlap F1,
not a new output parser tailored to the previous answers. The full decoded
response is scored: no extraction of a favorable final substring, removal of
reasoning, model-specific postprocessing, or retry of malformed responses.

## Data and freeze

Source: [official SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), dev-v1.1.json,
CC BY-SA4.0, SHA95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9.
Preparation code in `squad_coupling.py` chooses one eligible item per article
by a fixed hash, then8 articles by another fixed hash. Eligibility before any
model output:60–180-word passage, question<=30words, all annotated answers<=8words
and<=60characters.48 articles qualify. This bounds workload, not model difficulty.
All answer offsets are checked against the passage. No outcome-based selection.

Public cases and separate key under artifacts/squad_coupling_prepared_v1.
No locked TEST opened. This is old public DEV, not contamination-free evaluation
or evidence of 2026 model capability. Two small native model checkpoints and
tokenizer revisions match the earlier qualification; no new model downloads.

## Sampling matrix fixed before execution

-8 questions x16 fixed seeds x4 policies x2 models =1024 completions maximum.
-Policies: independent, token-clock/shared-token-byte noise, byte-clock noise,
  byte-clock/first-byte hierarchical noise. Their native marginals are the same
  by construction; finite-sample empirical means need not match.
-Full softmax at temperature1.0. No top-p/top-k/grammar constraints.32 native
  generated tokens maximum, EOS stopping. Native token caps are not byte caps;
  the target is each model's explicitly fixed sampler, not equal semantic budget.
-No answer-key access in generation, no oracle coupling groups.
-Batch8, CPU float32,2 threads, eager attention, native prefix KV caching.
  Cache/full-prefix and batch/single-sequence equivalence checks precede sampling.
  Maximum allowed native-model logit deviation5e-4 plus top1 agreement in the
  recorded probe. Synthetic Qwen/Llama unit tests check whole batched trajectories.
-Policy execution order rotates by question. Completed batch rows continue only
  as EOS padding; they emit no further recorded samples. Padding work is counted.

Sources, config, public inputs and model files freeze before the first forward.
Root artifacts/squad_coupling_dev_v1 must not exist at launch. Every output token,
sampling clock, selected-token log probability, raw decoded bytes and complete
answer is saved. Per-batch timing separates model and sampler work. No full-logit
archive for this larger DEV; record verification will not be called inference
replay. Exact rerun would require the pinned model forward passes.

## Fixed analysis and continuation criteria

For each policy, pair models by (question,seed). Report per-model EM/F1, EOS and
output lengths; per-question sample variance of F1_A-F1_B (and EM difference),
then the equally weighted mean across8 questions. This separates prompt difficulty
from the within-prompt generation variability the intervention targets.
Report all pairwise covariance terms and marginal variances, not only the ratio.
Sixteen seeds make variance estimates noisy; repeat samples are not new questions.

Report variance ratios to independent and both coupling baselines, and the product
of variance with mean measured paired generation cost. CPU timing is exploratory:
Python/NumPy overhead and batch padding are not optimized GPU wall-clock evidence.
Question-cluster bootstrap intervals are descriptive for8 DEV questions, not
population coverage claims. Shared seeds across policies do not imply independent
estimates; bootstrap paired question records rather than separate arm pools.

No paper green light from this DEV. Preparing a larger confirmation requires:
(1) byte/clock/cache checks pass; (2) neither model is at an answer floor or all
scores constant; (3) hierarchical coupling shows a practically useful reduction
over the *best* simple coupling baseline, not just independent; and (4) any gain
survives measured overhead. Use25% reduction as a planning target, not a p-value
or a threshold to tune against. A weak/mixed result parks this specific heuristic;
do not keep modifying groups, seeds, scoring or tasks to manufacture a win.

Even a clean positive would still require exact-byte sampler comparisons, multiple
current capable model families, fresh tasks, and a stronger novelty case. It would
not justify saying that the original models became more accurate or that a paper
is likely accepted. Preserve every result and failure; no automatic GPU expansion.
