# Shared-parameter Hindsight bifurcation follow-up

Frozen after the positive exact-map screen and before any LoRA outcomes. This is
the next mechanism gate, not a paper experiment selected by its result.

## Question

Does initialization-sensitive amplification survive finite optimizer steps and
shared neural parameters, or is it an artifact of independently and exactly
optimizing one Bernoulli policy per prompt?

The frozen Qwen3-0.6B base model receives rank-8 LoRA adapters. Development uses
only the 16 development contexts from the prior fresh-domain screen; confirmation
prompts are never optimized. Candidate sequence log likelihoods define the same
explicit two-action interface as the exact screen. For each bistable context, a
fixed context offset places the zero-adapter policy .01 below or above its measured
unstable point. This artificial local intervention tests stability; it is not a
claim that natural model initializations sit at that point.

Four arms start from the identical zero adapter:

- `dynamic_minus` and `dynamic_plus`: the report-one mixture weight is the current
  detached student probability after each update;
- `fixed_minus` and `fixed_plus`: that report mixture is frozen at the corresponding
  initial policy.

All arms use the same 64-update shuffled schedule, batch size8, AdamW learning
rate3e-4, rank8/alpha16, gradient clipping1, and candidate-normalized binary
reverse KL. Teacher distributions and offsets are fixed from the verified prior
artifact. Non-bistable development contexts are excluded rather than assigned a
post-hoc root. All confirmation contexts are retained for reporting.

The primary confirmation screen requires median dynamic plus-minus separation
at least.25, at least75% positive paired differences, and maximum absolute
fixed-arm separation at most.10. Failure parks the neural mechanism under this
setup. Passing warrants a larger, natural-initialization experiment; it still
does not establish user preference change, welfare loss, full-vocabulary SDPO,
or novelty beyond performative prediction.

Evidence root is fresh and immutable. Every adapter, optimizer, step record,
checkpoint evaluation, prerequisite hash and failure state is preserved. No
automatic expansion follows either result.

## Verified result

Frozen source commit `9d7551f`. All four arms completed 64 updates: 256 total
optimizer steps in 1,209.88 seconds on local CPU. The 14 bistable confirmation
contexts were never optimized.

The predeclared status is `LORA_BIFURCATION_SCREEN_NEGATIVE`. Preserve that
status: dynamic plus-minus separation is positive in14/14 confirmation contexts
and has frozen upper-median .999998, but the maximum absolute fixed-control
separation is .170681, exceeding the .10 ceiling. Three fixed pairs exceed .10.
The ordinary median fixed separation is .01719 (the runner's upper-middle order
statistic is .01963), compared with ordinary median dynamic separation .99981.

This is not a null finding. Dynamic feedback produced a dramatically larger and
direction-consistent split than the frozen-feedback arms, and it transferred to
unseen prompts. But the strict control shows that shared LoRA updates can amplify
the tiny initialization difference even without a changing report law on some
contexts. Therefore this run does not isolate all of the separation as a
performative feedback effect. The calibrated per-context offsets also make it a
local stability intervention, not evidence that natural initial policies suffer
the same collapse.

The next discriminating experiment, if pursued, must remove the calibrated
offsets and compare natural initial policies under dynamic versus frozen report
marginals, or use paired interventions whose fixed-control effect can be estimated
rather than bounded by a brittle maximum. That would be a new prospective gate;
the current threshold will not be changed after inspection.

Evidence: `artifacts/hindsight_bifurcation_lora_cpu_20260904_v1`; adjacent
verification receipt. All 20 manifest files, four adapters, four optimizer states,
256 step records, and 12 checkpoint evaluations are retained. Verification does
not deserialize the PyTorch states or replay neural forwards.
