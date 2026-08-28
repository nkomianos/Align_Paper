# Conditional candidate: shadow-student auditing for covert behavioral transfer

## Status

**Conditional G0 finalist; not a paper claim and not a safety certification.**
The proposed deployment action is only to *flag* a synthetic post-training batch
for review before a costly full run. A negative shadow result cannot establish
that a batch is safe, trusted, or free of an arbitrary undisclosed trait.

## Problem and hypothesis

Synthetic post-training data can transmit a behavioral trait that is absent from
its apparent semantics. Corpus inspection, model identity, and a first gradient
are not generally sound pre-flight tests: the carrier can sit in vocabulary
geometry or model-body computation rather than a removable surface feature.

The testable question is deliberately narrower:

> Given a declared behavioral probe battery, can a small, fixed ensemble of
> LoRA shadow students predict which synthetic data batches will cause a
> measurable behavioral shift in the corresponding full post-training run?

This is a compute-risk forecasting claim, not a universal trait-detection
theorem.

## Proposed method: SENTRY

For a candidate synthetic batch `D` and a prompt-, length-, and token-budget
matched neutral reference `D0`, train `k` fixed low-rank shadow adapters from
the deployment base. Measure a predeclared behavioral battery `B`, then
aggregate paired standardized drift:

```
S(D) = aggregate_b in B[(shadow(D, b) - shadow(D0, b)) / uncertainty_b]
```

SENTRY flags a batch only when the score's pre-registered lower confidence
bound crosses a calibrated threshold. It does not attribute a trait to a single
example, modify the batch, or certify a batch as safe. Deployment-scale effects
are measured separately by full post-training and sealed evaluation.

## Novelty boundary

| Closest work | What it establishes | Required distinction here |
| --- | --- | --- |
| [Subliminal Learning](https://doi.org/10.1038/s41586-026-10319-8) | Semantically scrubbed synthetic data can transfer traits. | Forecast deployment-scale transfer from fixed inexpensive post-training audits, rather than re-demonstrate it. |
| [Channel Location](https://arxiv.org/abs/2606.22019) | Initial-update screens are channel-dependent; post-hoc detection is needed, but its reported audit is not deployment-ready. | Use batch-level post-training shadows; test both vocabulary- and body-routed conditions, and never claim generality beyond the declared battery. |
| [Non-Semantic Distillation](https://arxiv.org/abs/2608.05734) | Gradients of *steered* data correlate with a known teacher steering vector. | Require neither the source intervention nor the teacher vector; forecast a full post-training outcome. |
| [Online Data Selection](https://arxiv.org/abs/2607.07023) | Online selection changes behavioral modes and ADA measures resulting drift. | Audit an incoming synthetic source before the full run, under covert-transfer controls, for compute-reduced forecasting. |
| [Small-run curation](https://arxiv.org/abs/2512.24503) and [small-to-large generalization](https://arxiv.org/abs/2505.16260) | Proxy-scale conclusions about data recipes can transfer, subject to recipe-specific tuning. | Target a conservative behavioral-risk flag under hidden transfer, defeat reduced-loss/fixed-config/data-statistic baselines, and include their reduced-learning-rate control. |

If the method merely detects source identity, tracks loss, or works only for a
calibration trait/channel, it is not differentiated and this candidate is dead.

## Frozen G0 contract

- Deployment base: Qwen3.5-9B at an immutable public revision.
- Deployment recipe: fixed LoRA targets, rank, learning rate, token budget,
  optimizer, and two independent seeds.
- Shadows: same base/tokenizer; rank 4; one eighth of the deployment token
  budget; four fixed adapter seeds; predeclared reduced-learning-rate control.
- Reference: every source batch receives a paired neutral batch with identical
  prompts, generation budget, token-length distribution, filtering, and train
  recipe. Source identity, order, and formatting must not predict the label.
- Challenge channels: at least one vocabulary-routed and one
  condition-present/body-routed trait, each through two public synthetic-data
  channels. The final channel/trait pair is locked before score calibration.
- Evaluation: public, fixed behavioral prompts with order-balanced answer
  formats, prompt-format controls, and no-claim controls. A single marker token
  or a prompt that shifts the neutral base is invalid evidence.

### Baselines

1. Initial update/coverage and update norm.
2. Corpus token-divergence frequency and perplexity/loss change.
3. One-shadow, fixed-learning-rate proxy.
4. Matched random retention.
5. Neutral paired source batch.

### Artifact binding before any GPU execution

The remote preflight must record, hash, and reject drift in all of the
following before the sealed evaluation file is made available to the runner:

1. The exact Qwen3.5-9B model revision and tokenizer revision.
2. The commit of the public [upstream Subliminal Learning
   repository](https://github.com/MinhxLe/subliminal-learning) and the
   SHA-256 digest of every downloaded public number/code positive-control
   artifact. Its published data are a positive control, not a sealed test set.
3. The commit of the independent Qwen2.5 offline replication, if used only to
   validate an input converter; its reported result cannot count as SENTRY
   evidence.
4. The deterministic generator code, seed, and emitted-data digest for every
   body-routed challenge/control pair.
5. A private answer key assigning calibration/sealed status, source identity,
   and expected signed behavioral direction. The runner may read assignments
   only after all artifacts above are attested.

`scripts/stage_sentry_public_sources_remote.sh` performs items 2--4's public
staging side only. It refuses an existing destination, pins all commits and
dataset revisions, records SHA-256 values, and intentionally does not download
the model or access the sealed answer key.

### Pass condition

On sealed held-out source batches, all of the following must hold:

1. The calibration and sealed sets each contain at least eight source batches,
   with at least four sealed batches in each channel and both positive and
   neutral full-run outcomes. The full run has a reproducible signed behavioral shift relative to neutral
   in both seeds for at least one held-out vocabulary-routed and one
   body-routed condition.
2. The frozen SENTRY score ranks full-run effect with macro Spearman >= 0.70
   and bootstrap lower confidence bound >= 0.50.
3. At a threshold calibrated only on non-held-out conditions, held-out recall is
   >= 80% at <= 20% FPR, with a lower confidence bound at least 10 points above
   every baseline.
4. Shadows consume <= 20% of full-run GPU time for the audited batch family.
5. No format-only, source-ID, or known-teacher-vector shortcut predicts the
   held-out labels equally well.

Failure of any condition kills the candidate. There is no threshold tuning,
channel deletion, or escalation after a G0 failure.

## If G0 passes

Require an independent model family and synthetic-data source, real benign
utility post-training alongside the controlled transfer challenge, and a
held-out operational review-queue evaluation (recall, workload, missed drift).
Only then would this be a credible ICLR submission candidate.

## Cost estimate

On one GH200, G0 should take roughly 4--8 GPU-hours for full runs plus under
2 GPU-hours for shadows/evaluation, after implementation preflight binds the
exact revision, public artifacts, and sealed test split. This is not yet
authorization to run.
