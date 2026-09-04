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

## Completed result — not a strong practical estimator split

Runner commit `085479e`; evidence root
`artifacts/hindsight_parameter_probe_cpu_v1_retry1`. The run completed all
66 forwards and eight backwards, applying one separate update per estimator.
Elapsed experimental time was **26.78 seconds on the laptop CPU**, excluding
the initial public-weight download. No rented GPU was used.

| Condition | Parameter-direction cosine, independently recomputed in float64 |
| --- | ---: |
| Independent feedback (rho=0) | 1.000000 |
| Primary action-copying channel (rho=.5) | 0.999156 |
| Stronger copying diagnostic (rho=.9) | 0.995805 |

| Model state | Train accuracy (8) | Heldout accuracy (8) | Heldout original-request NLL |
| --- | ---: | ---: | ---: |
| Before update | 4/8 | 6/8 | 0.724127 |
| Own-response one-step | 7/8 | 7/8 | 0.359969 |
| Full reverse-KL one-step | 7/8 | 7/8 | 0.356621 |

The independent-feedback parameter null is exact. Both one-step directions
are nearly parallel and improve this small test; there is no opposite-update
effect here. The mathematical sampling-law counterexample remains true, but
does not establish a practically important split in this setup. The low initial
training accuracy and eight-question heldout slice limit task-utility claims.
Do not turn either the improvement or the near-equality into a broad result.

The earlier saved-score audit used another model, other prompts and a different
feedback law. Multiple factors changed; this run does not identify which one
explains the difference. In particular, explicit original constraints may matter,
but no controlled ablation has established that attribution.

`scripts/verify_hindsight_parameter_probe.py` checks every artifact hash, source
pin, cases, labels, probability arithmetic, stored adapter differences and
gradient directions. Its separate report is
`artifacts/hindsight_parameter_probe_cpu_v1_verified.json`; no original artifact
was changed. It does not independently rerun model forwards/backwards.

Numerical audit: the runner's float32 reduction reported a null cosine slightly
above one. Recomputing from preserved directions in float64 gives the values
above. The actual update norms are 0.10000331 and 0.10000340, about 0.0034%
above the requested .1 from float32 norm reduction; the verifier confirms the
stored updates match their declared directions within that numerical tolerance.
These are explicitly reported numerical effects, not hidden by rerunning or
altering evidence. Teacher full-vocabulary A/B mass was not retained; only
teacher conditional A/B probabilities were saved. Student A/B mass is recorded.

45 relevant CPU tests pass, including chain-rule/direct-autograd agreement,
released loss execution and evidence-row validation. All failed-attempt and
successful artifacts and both actual updated adapters remain local. **No paper
green light and no expensive training expansion follows this result.** A next
diagnostic should isolate the changed factors before interpreting the earlier
logit-level sign reversals as neural-parameter or utility effects.
