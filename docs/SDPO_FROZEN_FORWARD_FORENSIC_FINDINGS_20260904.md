# Frozen-forward diagnostic: scalar findings and unresolved directions

## Evidence status

All 192 approved forwards completed. The diagnostic evidence manifest is `9836fa358b73f25a8c2bc08a0ed09a87ae50ee816ce3e5181262d53bd7cea5ed` under `retrieved/sdpo_frozen_forward_20260904T1007Z/sdpo_frozen_forward_20260904T1005Z`. Evidence was not modified.

The metricwise audit is `retrieved/sdpo_frozen_forward_20260904T1007Z/forensic_metricwise_v2.json`; the scalar-only summary is `scalar_summary_v2.json` beside it. Status is **PARTIAL_ARITHMETIC_REPLAY_DIRECTIONAL_METRICS_UNRESOLVED**, not fully verified arithmetic. File hashes, fixed selections, token contexts, checkpoint tensor receipts, and scalar/norm calculations passed their existing checks. No numerical tolerance was increased. Only the two saved vocabulary-logit positions support independently recomputed token log-probabilities. Other completion log-probabilities remain archived-forward-reported measurements, not independently replayed values.

## Why the first verifiers stopped

The original verifier returned an undefined cosine when the product of vector norms was at most `1e-15`; the runner returned a raw ratio for any nonzero product. There are 197 such reporting-convention discrepancies. For example, call 2's punctuation probe has vector norms approximately `1.13e-26` and `7.80e-14`: a displayed angle is numerically fragile despite mathematically nonzero vectors.

A separate v2 implementation preserved the original verifier and adopted the runner's raw-ratio convention. It also changed `A*p - A*onehot` to `-A*(onehot-p)`, the runner's centered-onehot operation order. These are algebraically identical but the latter avoids an unnecessary subtraction after multiplication near a probability of one. This change and the null convention correction were explicit forensic changes, not experiment changes.

Even then, 29 sampled-gradient cosine values did not reproduce within the unchanged tolerance. Some have one tiny vector and one large vector, so the norm-product criterion does not capture all instability. Call 68 has expected-gradient norm about `9.79e-12` and sampled-gradient norm `53.56`, with saved/recomputed cosine approximately `0.512820`/`0.513064`. The forensic report inventories each disagreement, the norms, and the independent analytic-versus-autograd error. It does not waive them. **Do not use raw cosine values or make directional claims from this diagnostic.** No replay of neural parameter gradients, optimizer dynamics, or training execution is claimed.

## Initially correct cases: feedback versus empty hindsight

The eight initially correct calibration cases belong to two plain-format users; they are not eight independent users or an external test set.

At the initial adapter, the mean probability of the original first token was:

| Context | Original first-token probability |
|---|---:|
| Base | 0.8304 |
| Empty hindsight block | 0.2864 |
| Actual feedback in hindsight block | 0.01383 |

Actual approval feedback lowers the original first-token log-probability relative to the empty block in all eight cases: mean difference `-3.167` nats, range `[-4.322, -1.682]`. Thus the empty wrapper itself is a substantial perturbation, and the actual feedback adds another substantial perturbation. This is a distributional observation, not yet a semantic-harm metric.

The already archived free-generation calibration provides an important separate check: on these eight cases, the feedback teacher preserves strict format success in `0/8`, while explicit-preference calibration succeeds in `8/8`; both preserve grounded content in `8/8`. One example changes from `Asset: ...` / `Zone: ...` to a one-line `asset = ...; zone = ...` response under approval, violating the frozen plain-format rule. Explicit calibration instead emits valid lowercase `asset: ...` / `zone: ...` lines.

That explicit example also shows why low probability of the original first token is **not** low probability of correctness: lowercase `asset` can replace uppercase `Asset` harmlessly. A concrete checked pair is `calibration/user_89d2999b9c0f/v3-0`: original `Asset: ...` / `Zone: ...`, explicit `asset: ...` / `zone: ...`, both strict-joint correct. This valid-alternative control is central to interpretation, not a minor caveat. We did not generate an empty-block response, so we cannot attribute the teacher's entire format error specifically to approval rather than the wrapper, or quantify preservation under the empty control.

## Checkpoint changes and local magnitudes

For the initially correct cases, base probability of the old first token is about `2.42e-11` at step 16 and `7.37e-12` at final. Feedback-minus-empty first-token log-probability differences become positive (`+0.359` and `+0.500` nats), but both remain far from the old first-token distribution. Do not describe this as recovery: these are frozen-prefix evaluations and the original final free-generation result remains unchanged.

At initialization, the correct-case first-position fixed-feedback gradient expectation norm averages `0.923`, with conditional trace variance `18.58`; the empty-block counterparts are `0.471` and `0.370`. These are categorical local-logit quantities holding the selected feedback and prefix fixed. They do not account for feedback changing when another full response is sampled. They therefore do not establish an unbiased full SDPO estimator or identify parameter-update noise.

For the selected initially wrong cases, initial feedback first-position reverse KL averages `34.84`. At step 16 it is still `28.58`, despite a local expectation norm around `3.79e-6`; at final these are `22.53` and `2.08e-6`. This is consistent with saturated local distributions: large distributional disagreement need not yield a large local-logit gradient. It is not a unique diagnosis of the training failure.

Released top20-plus-tail and full-vocabulary reverse-KL scalar values and gradient magnitudes are close at the inspected positions. For example, on initially correct first-position feedback probes the mean gradient L2 difference is about `5.64e-8` initially. This offers no evidence that top20 support omission explains this specific diagnostic. It does **not** imply that replacing sampled updates with full distillation would or would not fix training; that would require another controlled intervention.

## Review of the single-profile reproduction proposal

A closer released-recipe reproduction is scientifically reasonable as an apparatus check, not as a paper result or a causal isolation of the old failure. Preserve the explicit statement that simultaneous changes to profile allocation, loss, LR, and adapter capacity are bundled.

The proposed minimum four satisfactory and four unsatisfactory calibration cases is fair only as a requirement for estimating both preservation and recovery; failure to obtain those strata means insufficient calibration coverage, not failure of SDPO. On four cases, an 80% preservation rule requires all four to succeed; report the integer counts. Likewise, 75% recovery is a coarse small-sample engineering threshold, not convincing inference.

Define content corruption prospectively as material factual error against the source, with a fixed rubric; avoid rejecting harmless paraphrases. The same simulator serving as judge remains circular external evidence even if evaluation order is balanced. Finally, a promise of **human** inspection requires an actual human reviewer; another model's inspection must be labeled model-assisted review. None of these qualifications justifies further parser repairs or a large hyperparameter search.

## PI interpretation

The evidence supports a concrete **positive-control defect**: the initial feedback-conditioned teacher fails to preserve already satisfactory strict-format behavior, and both the hindsight wrapper and feedback alter its token distribution. It also shows subsequent local saturation. It does not identify one cause of collapse or disprove the original causal-identification idea. A closer natural-data reproduction may establish a usable learner, but it is not yet evidence for a novel paper.
