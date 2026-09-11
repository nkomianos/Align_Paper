# Calibration failure is not isolated to held-out transfer

Reran verify_hindsight_calibration.py on the retrieved sealed calibration and
hash-pinned learning.json. Raw saved-logit arithmetic, manifest, row bindings,
training schedule and routing verify. This is not neural checkpoint re-execution.
Route remains INVESTIGATE_DISTILLATION_OBJECTIVE_OR_TRANSFER; no paper green light.

| Arm | Training accuracy | Held-out accuracy | Training position range | Held-out position range |
|---|---:|---:|---:|---:|
| Supervised |.380859|.335938|.045941|.044196|
| Frozen teacher |.294922|.320313|.236587|.268486|
| Updating teacher |.250000|.250000|.998859|.999525|

The final training scores already show weak acquisition and, for distillation,
position dependence. Thus the endpoint does not support an explanation confined
to held-out transfer. Frozen-teacher loss improvement is real; the failed.20
position-range criterion is not equivalent to no learning. SFT passed a modest
improvement gate, not high-accuracy task mastery. Updating-teacher label collapse
appears in both splits. No internal causal mechanism is identified by this table.

The sampled learning population was previously exposed.128training bases and
32held-out bases each have4dependent answer rotations;512/128rows are not those
many independent tasks. The32optimizer steps and prospectively frozen threshold
are preserved. No longer training, label debiasing, objective change or learning
rate rescue is retroactively counted as tested.

Disposition: retain an apparatus/recipe failure, not a negative for the full
causal Hindsight correction, whose downstream method comparison was not admitted.
A new qualified learning setup remains necessary, but fixing acquisition alone
would not resolve the separate novelty and latent-measurement limitations.
No GPU run admitted from this recheck.

Raw root: retrieved/lambda_h100_20260909/calibration/suite_v3/hindsight_calibration.
Learning root: retrieved/lambda_h100_20260909/learning_inputs/learning.json.
