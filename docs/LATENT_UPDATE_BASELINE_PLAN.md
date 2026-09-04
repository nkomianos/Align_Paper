# Latent-interface update study: conditional baseline plan

Status: released-C2C inference completed and secured; post-hoc scoring diagnosis
shows a useful interface, but the original strict-format gate failed. No sender
update trained and no paper-level go. The nonce-channel DEV was a separate
apparatus test, not a published-method reproduction or novelty result.

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

### Completed CPU/data preflight, 2026-09-04

`scripts/prepare_c2c_baseline_dev.py` prepared 64 OpenBookQA and 64 ARC-Challenge
validation examples, selected by a fixed salted hash of source ID. OpenBookQA
has 500 eligible validation examples; ARC-Challenge has 291 of 299 with four
source labels in ABCD order. The latter restriction matches the audited upstream
formatter/scorer contract, not answer correctness. Selection is independent of
labels and row order. No TEST split was downloaded. This publicly available DEV
slice cannot support claims of new uncontaminated held-out generalization.

- Prepared root: `artifacts/c2c_baseline_dev_20260904_v1`.
- Public cases SHA-256: `a279178264c7b2e66c65d852193723925b42482d532ef6dc568a5bf3d0ce046b`.
- Manifest SHA-256: `7104a4165454138c6ca24099bcd6dc8f8f470b228b2b43590c03eefe1aa2be87`.
- OpenBookQA revision: `388097ea7776314e93a529163e0fea805b8a6454`.
- ARC revision: `210d026faf9955653af8916fad021475a3f00453`.

`scripts/inspect_c2c_projector_cpu.py` downloaded only projector 0, checked its
released LFS SHA-256, loaded every state-dict key strictly, and ran deterministic
finite synthetic-tensor forwards on CPU. Result root:
`artifacts/c2c_projector_cpu_20260904T0452Z`. Weight digest:
`f60f3b3e5fd27a96cee8d9d30de8bfdd2a88bc041ae70ade8cb9e29c06d5e4a8`.
This used local torch 2.11.0+cpu / Transformers 5.6.2, not the published full-model
environment. It tests one projection module only: no receiver or sender model
loaded, no natural-language evaluation, and no inference-time cache hooks tested.
The key gate is closed and value gate open for this layer in eval mode; a closed
learned gate is not a loading failure. Full baseline inference subsequently
completed; see the README and journal for results and the separate scoring audit.

### Frozen inference baseline (now completed; original contract preserved)

`scripts/run_c2c_baseline_dev.py` and `scripts/verify_c2c_baseline_dev.py` define
512 calls: the 128 prepared questions in receiver-only, sender-only, released
C2C, and disabled-fuser arms. The same upstream prompt builder, non-thinking
chat template and 64-token greedy generation cap apply. Require identical
input tokens across both model tokenizers before any generation. Every fuser
loads strictly; inference uses the official unmodified C2C wrapper at the pinned
source commit. A separate isolated ARM64 environment uses Transformers 4.52.4
and CUDA torch 2.7.1, explicitly not exact upstream torch 2.6.0 reproduction.

The strict answer parser accepts a lone A/B/C/D or `The correct answer is X`
with an optional final period; it does not recover letters from explanations.
Functional prerequisites are >=95% parse rate per arm and >=98% text agreement
between receiver and disabled-fuser outputs. Conditional on these, a >=5pp
C2C advantage only advances protocol design. No training, text-transfer arm,
novelty claim or paper go is included in this baseline. Other outcomes are
inconclusive for this release/task slice, not a refutation of C2C.

All prompts, tokens, timings, asset hashes, runtime and source are preserved;
analysis uses the private key locally. Run only after the active 96-call DEV
process has exited and its evidence has been secured. This baseline is relevant
regardless of the custom nonce assay result because the two interfaces differ.
No automatic update training follows it.

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
The [Stage A training/qualification implementation](C2C_NATURAL_UPDATE_PILOT.md)
now exists and has CPU tests. The matched text comparator and paired interface
runner are implemented but unrun; repair comparisons still need implementation.

## Novelty boundary

[Lambda-Orthogonality](https://arxiv.org/abs/2509.16664) already addresses
compatible updated representations using regularized alignment in retrieval.
[StateBridge](https://arxiv.org/abs/2608.13317) already addresses training-free
hidden-state alignment. [Latent Cache Flow](https://arxiv.org/abs/2605.22863)
already improves cache communication. Generic drift, alignment, or cache
efficiency is not enough. A bounded search has not established priority for the
specific deployment-update claim; absence from search is not novelty proof.

### Additional collision check before update training

[Heo et al., August 2026](https://arxiv.org/abs/2608.03893) already study
within-family cross-model prefix reuse with closed-form ridge mappings and
unlabeled calibration. [CacheBridge, September 1](https://arxiv.org/abs/2609.00891)
adds head-local structure, attention-sensitivity weighting and faster fitting.
Thus neither plain ridge repair nor attention-weighted residual fitting should
be presented as our new method. Include these baselines where the interface
permits, while distinguishing complete prefix-cache replacement from C2C fusion.

CacheBridge's stated limitations include open-ended multi-turn continuation,
unmatched attention layouts and cross-family transfer. These are possible
research openings, not automatic novelty: merely evaluating another benchmark
or recording accumulated error is unlikely sufficient. An update study still
needs natural independent updates and a non-obvious result beyond adapter
incompatibility. No additional GPU study is authorized by a literature gap alone.
