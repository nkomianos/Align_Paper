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
