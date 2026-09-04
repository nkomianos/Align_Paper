# CPU parameter-gradient probe v1 — frozen before execution

This is a one-step diagnostic on pretrained Qwen3-0.6B, not a training campaign,
replication of the upstream SDPO results, or submission-ready experiment. It
asks whether the previously derived estimator difference survives aggregation
through shared trainable parameters, and measures immediate task changes.

## Frozen choices

- Public model revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Laptop CPU, float32, eager attention, four threads, seed 9046201.
- Rank-4 / alpha-8 LoRA on all attention q/k/v/o projections. Base weights frozen;
  zero B initialization. Check exact equality before and after adapter insertion.
- Four original-request tasks (storage, travel, appointments, purchases), each
  in two option orders: eight training contexts. Four fresh analogous scenarios,
  also order-balanced: eight heldout contexts. Handwritten development data,
  no broad natural-language/generalization claim.
- Restricted categorical A/B policy, no generation or EOS learning. Record A/B
  full-vocabulary mass; do not confuse renormalized probabilities with a
  full-vocabulary update. Maximum 256 input tokens, no truncation.
- Exact pinned upstream hindsight block in a single user message. Feedback
  says it prefers one option. The feedback law is designed: with probability
  rho it copies the action, otherwise it states the original correct option
  with probability .9. This is expression corruption, not measured persuasion.
- Compute exact expected gradients for rho 0, .5 and .9, with stopped teacher
  and sampling weights. No sampled-action variance. Rho=0 is the
  action-independent null. The actual one-step comparison uses rho=.5 only.
- Compare own-response scoring against full reverse-KL directions in the
  adapter parameter space. Both directions are applied from the same initial
  adapter with L2 step norm .1. This isolates direction, not original loss
  scale or AdamW dynamics. Do not call it ordinary SDPO training.
- Save initial adapters, both gradient directions for all strengths, and both
  one-step adapters. Evaluate train and heldout original-request accuracy,
  restricted cross-entropy and A/B mass before and after each step.

The calculation uses the chain rule: each context's restricted binary policy
has one log-odds Jacobian. Multiplying that Jacobian by the exact scalar ascent
coefficient yields its parameter direction; averaging across contexts can change
alignment. The implementation is checked against direct shared-parameter
autograd. The original-task log-likelihood direction is only an oracle audit
reference, not a proposed deployable learner.

Planned computation: 66 forwards (including two no-op controls), eight
log-odds-Jacobian backwards, and two distinct one-step adapter updates. No
optimizer sweep, checkpoint selection, automatic extra training or GPU use.

## Interpretation fixed before results

Report all strengths and both one-step outcomes. If the null fails, treat the
implementation as invalid. Opposite parameter directions are not guaranteed
by the earlier per-context result. A meaningful mismatch is a mechanism clue,
not evidence that one estimator is useful. Near-identical directions weaken
the motivation for this exact setup.

Low baseline task accuracy or low A/B mass limits utility interpretation. A
single step's loss reduction cannot establish stable learning, safer preference
adaptation or heldout robustness. The tiny paired scenarios do not support a
paper acceptance claim regardless of the numerical outcome. Preserve negative
results; do not retune the feedback law or step size on them.

Code: `src/interaction_sprint/parameter_probe.py`; data and settings are emitted
into a fresh root before model loading. Only already downloaded public files
are loaded. Model download is separate; no external service/API is invoked by
the experimental runner.

Pre-inference engineering note: the first launch stopped before any scientific
forward because Transformers 5 returned a BatchEncoding by default. Explicit
`return_dict=False` restores the intended token-ID list. Preserve the failed
root `artifacts/hindsight_parameter_probe_cpu_v1`; the retry uses a fresh root.
No data, prompt content, model, objective or decision rule changed.
