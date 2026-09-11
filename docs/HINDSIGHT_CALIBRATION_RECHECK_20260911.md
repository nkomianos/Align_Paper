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

## Equal-label retrieval check, September 11

The actual student prompt contains a displayed name, a purchase request and
four options; it does not contain the teacher's delayed preference follow-up.
The 128 training bases have 20 display-name groups. The 32 held-out bases have
19 groups, all also present in training. These are not verified person identities
or a guarantee of a stable preference state across source records. The protocol
already disclaims unseen-user evidence; this check does not discover new leakage.

Two CPU retrieval rules used the same 128 old-target training labels. All four
rotations were checked to have identical semantic target text and collapsed to
one base. TF-IDF vocabulary was fit on training text only; predictions were
formed without held-out gold. Lexical ties, not answer-letter positions, resolved
equal candidate scores. Each comparison uses 32 base tasks, not 128 rotations.

| Exploratory rule | Same displayed name | Fixed cyclic other name | All names |
|---|---:|---:|---:|
| Mean profile of training target texts | 6/32 | 2/32 | 1/32 |
| Nearest training task, then target-to-option match | 8/32 | 12/32 | 11/32 |

The first rule selected the generic none-of-these option on 20, 17 and 32 tasks,
respectively. It is a poor baseline, not evidence that personalization is
impossible. The second was explicitly designed after inspecting that failure;
it is posthoc development, not independent confirmation. It selected none-of-
these on 5, 2 and 1 tasks. No parameter sweep or neural call was performed.
Neither comparison establishes a same-name retrieval advantage. Display-name
matching may also mix distinct source preference states, so failure does not
prove that actual personal history lacks useful information.

These results do not qualify a replacement for the failed distillation setup,
and do not establish a universal learnability ceiling. The pooled nearest-task
score of 11/32 is merely descriptive; it must not be called a significant win
over the SFT arm's rotation-averaged 0.335938 accuracy. All variants use exposed
developmental data and are dependent analyses, not multiple replications.

Sources: `scripts/audit_hindsight_profile_retrieval.py` (frozen at f1dd75b before
execution) and `scripts/audit_hindsight_nearest_task.py` (posthoc variant).
Raw artifacts: `artifacts/hindsight_profile_retrieval_20260911.json` and
`artifacts/hindsight_nearest_task_20260911.json`. Both bind the sealed INPUTS.json
SHA-256 44730ee2f08f84f71b4cbca834820fd0485793facdaebfdc82a24a151ba0de8a and
their executed source hashes. The input manifest was verified before each run.

Hindsight remains parked. A revival needs an observable, learnable deployment
task with correctly bound persistent-state targets, not merely more optimizer
steps or explicit insertion of the hidden target into the student's context.
That latter shortcut can remove the identification problem the paper is meant
to study. Acquisition, sparse-label advantage and external persistence validity
still require separate evidence.

## Saved trajectory and uniform-choice control

Authenticated all ten saved held-out score files (baseline and steps 8/16/32
for three arms), checked identical 32-base/4-rotation coverage, and verified
per-row NLL = -log(conditional correct probability) - log(choice mass), within
2e-6 numerical tolerance. This replays saved score fields, not new raw logits or
neural checkpoints. Artifact: `artifacts/hindsight_loss_decomposition_20260911.json`;
source: `scripts/audit_hindsight_loss_decomposition.py`.

| Arm | Conditional NLL, step 8 | Step 16 | Step 32 |
|---|---:|---:|---:|
| Supervised | 1.246032 | 1.298546 | 1.249061 |
| Frozen teacher | 1.515465 | 1.457462 | 1.460350 |
| Updating teacher | 1.510237 | 1.562289 | 8.236793 |

Initial conditional NLL was 2.036798. At the final SFT endpoint, its reduction
was 0.787737, whereas the reduction in -log(choice mass) was only 0.003422.
Thus its loss improvement is not merely increased mass on the allowed answer
tokens. This does not establish semantic mastery: changes in within-choice
bias/calibration also affect conditional NLL.

A uniform four-choice predictor has conditional NLL log(4)=1.386294 and
expected accuracy 0.25. Relative to the initial model it gains 0.650504 NLL;
the final SFT arm improves a further 0.137233 beyond uniform. The former is
82.58% of SFT's total conditional-NLL improvement. That arithmetic comparison
does not attribute 82.58% of the optimizer's actual mechanism to debiasing.
Frozen-teacher final conditional NLL is worse than uniform despite its higher
argmax accuracy; the two metrics assess different properties.

SFT's step-32 conditional loss is essentially its step-8 level. Updating-teacher
deterioration is concentrated after step 16 in the available checkpoints.
Neither observation proves convergence, identifies an internal mechanism, or
rules out every longer-training recipe. It does rule out presenting the saved
trajectory as steady evidence that more of the identical recipe is likely to
qualify. No checkpoint is retrospectively substituted for the fixed step-32
endpoint. No GPU extension or downstream correction comparison is admitted.
