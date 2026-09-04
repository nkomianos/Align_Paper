# Fresh preference control using the released hindsight block

Freeze before execution: eight new ordinary preference domains (16 option-order
contexts), both possible target preferences, no overlap with prior probe domains.
Model and CPU execution settings unchanged. No training, sampling or filtering.

Three target-conditioned teacher variants:

1. Published: append the exact `hindsight_block_template` extracted without
   executing code from the hash-checked upstream config at commit
   `3b17d2a67bd2565b9fbda495fd16a485406aa954`.
2. Direct: append the same preference statement directly to the user request.
3. Redacted: the diagnostic redacted-answer wrapper chosen on earlier cases.

All 32 pairs per arm plus 16 no-preference base prompts and 16 published-block
generic-thanks controls = 128 forwards. The published arm deliberately does not
add the old assistant answer to the prompt. This replicates the single-user
message append operation, not the full teacher-forced multistep SDPO loss or
optimizer. Other settings of the released training pipeline are not reproduced.

Report every domain, A/B order, correct-target probability, A/B vocabulary mass,
paired recoveries/regressions against direct, and label bias of uninformative
controls. This tests whether our tiny model and prompt can represent the learning
signal before drawing any conclusion about learning from endogenous feedback.
No one expects a model to know a hidden individual preference without feedback:
base predictions are priors, not wrong-user conclusions. Fresh cases are a
developmental transfer check, not a large held-out paper benchmark.

Any apparent teacher improvement is an apparatus result. A useful Hindsight paper
still requires actual learning, identifiable evaluation objectives, strong
same-information baselines and a substantive novel result.

## Completed: fresh-case transfer, not a paper result

Frozen commit `c4016fb`. Completed 128 CPU forwards in 47.61 seconds with no
updates. Saved evidence and exact prompt construction verify; no neural replay
was performed for this new run. Minimum A/B probability mass is 0.9999839.

| Teacher input | Correct / 32 | Mean true-option probability | Correct when target A / B (each /16) |
| --- | ---: | ---: | --- |
| Released hindsight block | 25 | .77179 | 16 / 9 |
| Direct preference | 27 | .82982 | 16 / 11 |
| Redacted diagnostic wrapper | 30 | .89447 | 16 / 14 |

Relative to direct preference, released hindsight has two regressions and no
recoveries. Redacted has three recoveries and no regressions. This repeats the
direction of the earlier redaction advantage on fresh domains but still involves
only eight domains and one very small model; do not infer robust superiority.

Both no-preference base and generic-thanks controls choose A in all 16 cases.
Mean A probability is .99701 and .96540 respectively. Since the true preference
is absent, these controls do not test individual-preference accuracy, but they
expose a severe default label bias that any learning study must report and
counterbalance. Adding generic thanks alone changes the prior without revealing
which alternative the simulated person actually prefers.

The released template does convey useful preference information. It does not
eliminate the option-order weakness of this small-model apparatus. The redacted
variant is one additional teacher baseline, not an established new algorithm.
The next useful experiment is actual matched learning with truthful and
response-dependent feedback, preserving option-order pairs and comparing to
direct preference supervision with the same label budget. More prompt-only
qualification runs would not establish the user's requested learning claim.

Root: `artifacts/hindsight_published_control_cpu_20260904_v1`.
Receipt: `artifacts/hindsight_published_control_cpu_20260904_v1_verified.json`.
All unique inputs, outputs, source and runtime records are retained. Process
exited successfully; no GPU is active.
