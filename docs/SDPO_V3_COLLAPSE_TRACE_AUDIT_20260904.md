# SDPO v3 collapse: saved-trace audit

Read-only audit of `retrieved/sdpo_v3_20260904T0945Z/sdpo_format_v3_20260904T0915Z`. No regenerated predictions, repairs, reruns or threshold changes. The separate verifier checks saved evidence, not neural-forward or optimizer replay.

## Observations more specific than a routing explanation

The format failure emerged after just six updates. Training steps6–63, all58 remaining samples, use `asset = VALUE; zone = VALUE.`. All64 final held-out outputs use that same layout, independent of user. This is a shared output-mode collapse, not just an aggregate score concealing successful personalization for some users.

Only training steps4 and5 satisfy their assigned strict format. Both receive truthful generic approval. Nevertheless:

- Step4: correct plain answer starts `Asset`; first-token base log probability is−0.31343, teacher−9.75008, advantage−9.43665. Its pre-clipping gradient norm is22.2599.
- Step5: correct plain answer starts `asset:`. For the colon token, base log probability is−0.0232454, teacher−29.0, advantage−28.97675. Its pre-clipping gradient norm is19.1518. The next sampled answer uses equals signs and a semicolon.

Thus this run's actual update signal strongly penalizes a correct format immediately after positive feedback. This is not a claim that the optimizer sign is reversed: the teacher itself assigns the correct punctuation much lower probability. Generic approval does not tell the hindsight teacher which of the four layouts was approved; the original completed response is not included in its prompt. The teacher is scored on sampled prefixes, but at a layout decision it need not infer the correct user-specific target from approval.

The initial calibration required mismatch recovery, not preservation of already-correct outputs. In fact all24 initial mismatches recover, whereas total teacher joint success is24/32. That implies the other eight originally correct cases all lose strict success under the hindsight prompt. This warning was present before training but was outside the frozen qualification rule. We should acknowledge that design omission rather than call the training result a clean test of a beneficial-feedback learner.

## Why later negative feedback did not immediately recover

At step6, the collapsed first token has base log probability−0.000004649, while the teacher assigns−37.875. At step12 the wrong equals-sign token has base log probability0 at saved precision and teacher−47.9375. The feedback explicitly asks for colons, so the teacher still rejects that wrong punctuation.

Despite those large log-ratio penalties, every step6–63 pre-clipping gradient norm is below0.000711; absolute losses are below0.000007748. For a sampled-token stopped-advantage loss, the log-softmax gradient becomes small when the sampled action already has probability nearly one. Large disagreement in teacher log probability does not itself supply a large student score-function gradient. Early large updates, the shared adapter, optimizer momentum and policy saturation are compatible with this trace. Their relative causal contributions cannot be isolated from these saved trajectories alone.

## Bridge/loss audit

The archived loss is the pinned upstream `simple_signal` implementation: negative mean of stopped `(teacher_logp - base_logp)` times base log probability. Negative advantage therefore penalizes the sampled token under gradient descent; no reversed sign was found. The archived independent float64 reference has zero loss and gradient error, with no teacher gradient.

The bridge forms `[context, completion[:-1]]`, retains the last `len(completion)` logits, and scores exactly `completion`. The first retained logit is the final context position, which predicts the first response token; the last predicts the final emitted token. It uses native IDs, actual EOS only, no padding and an all-one completion mask. Teacher/base contexts differ only by the intended hindsight block. This is correct next-token indexing, not a one-token shift.

Across all64 sampled trajectories, the maximum generation-versus-full-prefix selected-token log-probability gap is approximately0.0000100132. This strongly checks base-side alignment, though it does not independently replay teacher neural forwards. The verifier independently reproduces saved loss arithmetic and binds archived sources. Saved adapters change and optimizer state reaches64 steps. There is no concrete evidence here for an implementation sign, masking or indexing bug; this is not an exhaustive proof of every numerical operation.

## PI interpretation

The narrowly supported result is **early shared-format collapse with an adverse hindsight signal even on truthful approval, followed by saturated sampled-token gradients**. Routing interference is plausible but not sufficient as the diagnosis, and the trace does not justify calling this an unavoidable SDPO failure.

The positive-control criterion fails as recorded. Do not proceed to endogenous-feedback arms. A future method-faithful positive control would need to qualify preservation on already-correct examples as well as recovery, and separate single-profile adaptation from arbitrary user-ID routing. Such work would require a new prospective design; this audit does not authorize a repair or additional training.
