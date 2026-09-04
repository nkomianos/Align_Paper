# Latent-interface update study: conditional baseline plan

Status: design and provenance preflight only. No sender update trained, no C2C
weight loaded, no paper-level go. Current nonce-channel DEV is a separate apparatus
test; passing it does not reproduce a published method or establish novelty.

## Claim worth testing, not an assumed result

Does ordinary sender fine-tuning damage a frozen latent communication interface
more than text communication, despite retained sender task performance? Can a
small unlabeled compatibility repair recover held-out performance at less cost
than rebuilding the bridge or falling back to text?

Representation drift itself is not novel. An arbitrary hidden-space rotation is
not a realistic update and cannot establish the main result. A successful paper
would need repeatable failures under independently chosen useful updates, a
well-controlled explanation, and a correction beating existing alignment and
operational fallbacks. If all we find is a broken adapter after changing its
input model, the contribution is probably too obvious.

## Published baseline and preflight

- Official [C2C source](https://github.com/thu-nics/C2C), audited commit
  `113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a`.
- [Released fuser](https://huggingface.co/nics-efc/C2C_Fuser), revision
  `f01fc3258b305e280e04c7238f4f2cf31b7dc70d`, selected directory
  `qwen3_0.6b+qwen3_4b_Fuser/final`.
- Receiver Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`;
  sender Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
  Upstream config names repositories without revisions: these are our pinned
  snapshots, not evidence that they exactly match the original evaluation.
- Released config freezes both models, trains fusers on 500,000 OpenHermes
  examples, and maps sender layers 8–35 to receiver layers 0–27. This is a
  cache-fusion baseline, not the hidden-prefix method in our current DEV run.
- Do not install upstream torch 2.6.0 / Transformers 4.52.4 into existing envs.
  Establish an isolated compatible ARM64 environment and record any deviations.
- Upstream loader permits missing/unexpected state-dict keys (`strict=False`).
  Our reproduction must inspect and reject unexplained mismatches, use pinned
  local snapshots and safe tensor loading, and disable external telemetry.
- Preserve upstream LICENSE. Apache-2.0 LICENSE and MIT package metadata conflict;
  do not redistribute an adaptation without resolving licensing scope.

`scripts/audit_c2c_release.py` saves release metadata, model configs, exact file
inventory and available LFS digests without loading weights or executing upstream
code. It does not certify model behavior or environment compatibility.

## Before paying for update training

1. Reproduce inference with the released fuser on a predetermined published-task
   DEV slice. Record receiver alone, sender alone, text transfer, C2C, and
   disabled-fuser outputs. Preserve exact prompts, decoding budgets, token IDs,
   cache mapping, and end-to-end latency. Inspect the official evaluation code
   before freezing task/format choices; do not substitute our nonce task.
2. Require a usable C2C baseline versus the unfused receiver. A broken loading
   path or nonworking baseline is an engineering failure, not an update effect.
3. Freeze disjoint update-training, update-qualification, repair-calibration and
   final evaluation sets. Published fuser training contamination remains a
   limitation; final claims need independently constructed or held-out tasks.
4. Select useful update recipes/checkpoints using sender-only DEV performance,
   never their eventual bridge degradation. Include no-op save/load and two
   independently seeded ordinary LoRA updates. Specify all learning parameters,
   seeds and checkpoint selection before observing update-channel outcomes.

## Required paired controls and estimands

For each fixed question, evaluate old/new sender with frozen receiver and bridge.
Report sender-only accuracy, output-distribution changes, text-transfer accuracy,
C2C accuracy, formatting failures and latency. Do not treat two arms on the same
question as independent samples. Questions and independently trained updates
are different sources of uncertainty; two seeds do not support a broad
population claim about updates.

Primary descriptive contrast: (new minus old C2C accuracy) minus (new minus old
text-transfer accuracy), with paired intervals. Also report both unadjusted
changes. This contrast does not, by itself, causally isolate geometric drift:
text contents and semantic capabilities may change differently across channels.
Use prespecified sender-behavior qualification and representational diagnostics;
do not claim a mechanism from a difference-in-differences number alone.

Repair comparisons: no repair, diagonal calibration, ridge/orthogonal alignment,
small fuser retuning, sender/receiver version checks with text fallback, and
full fuser retraining if warranted. Account for the cost of keeping the old
sender and generating calibration caches. No supposedly free oracle repair.

Provisional advancement target, to be frozen before training: repeatable >=10pp
extra communication loss with sender accuracy within a prespecified retention
margin, and recovery of >=75% of loss on held-out tasks with an end-to-end
efficiency advantage. These thresholds do not ensure novelty or acceptance.
Training is not launch-ready until the previous items and protocol are complete.

## Novelty boundary

[Lambda-Orthogonality](https://arxiv.org/abs/2509.16664) already addresses
compatible updated representations using regularized alignment in retrieval.
[StateBridge](https://arxiv.org/abs/2608.13317) already addresses training-free
hidden-state alignment. [Latent Cache Flow](https://arxiv.org/abs/2605.22863)
already improves cache communication. Generic drift, alignment, or cache
efficiency is not enough. A bounded search has not established priority for the
specific deployment-update claim; absence from search is not novelty proof.
