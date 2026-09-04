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
