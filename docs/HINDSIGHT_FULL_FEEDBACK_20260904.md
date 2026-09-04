# Full-vocabulary feedback learning — prospective staged comparison

Reuse the qualified acquisition dataset/schedule:64 training rows (four phrasings,
eight domains, paired option orders),32 evaluation rows (two other phrasings),
rank8/alpha16 attention LoRA,96 AdamW steps, batch8, lr.0003, clip1, no weight
decay. Each arm resets to the saved **initial**, zero-update acquisition adapter;
it does not inherit the trained oracle's preference knowledge. Preserve the
previous experiments unchanged. This is developmental personalization, not an
independent benchmark or a human-user study.

## Full-vocabulary and invalid-action contract

Freeze base-model full-vocabulary log probabilities for three feedback messages
per training prompt, using the pinned upstream hindsight block:

- Report preference for option A.
- Report preference for option B.
- Request an A/B answer after a different first token.

Let pA/pB be **unconditional** student token probabilities. With probability
1-rho the simulator reveals the true preference; with rho it copies a valid
sampled A/B response, or requests a valid format for any other token. Thus the
three report probabilities are
`rho * [pA, pB, 1-pA-pB] + (1-rho) * one_hot(true option)`.
This avoids silently renormalizing a low-mass A/B policy. It also means rho arms
receive a formatting signal on invalid actions, which must be disclosed rather
than interpreted as a pure preference-only effect. Full-vocabulary output and
A/B mass are reported separately.

Train with the exact report-marginal average of full-vocabulary reverse KL.
Report weights and teacher are stopped; no top-k approximation. This is a
frozen-teacher, one-token, known-simulator marginal objective, **not** a complete
adaptive-teacher SDPO reproduction or same-rollout sampled estimator.

## Ordered arms and stopping rule

First train truthful-feedback KL (rho0). Evaluate only its final checkpoint.
Require the same acquisition criteria: >=90% full-vocabulary argmax accuracy
on32 evaluation rows, >=75% each domain, all A/B masses >=.95. If it fails,
save everything and stop: remaining costly arms are not an informative test of
the hypothesis until truthful hindsight learning works.

If it qualifies, run from the same initial weights and fresh optimizer:

1. Copying KL rho.9.
2. Initial-policy fixed-marginal noisy KL, matched to arm1 at initialization.
3. Copying KL plus anchors (batch-mean KL + mean available-anchor NLL).
4. Anchor-only, identical anchor exposures and96 optimizer steps; zero gradient
   on batches without anchors, with Adam momentum retained.
5. Direct all-label supervision as an execution/positive-control replication.

Anchors retain the previously seeded two domains,16 distinct training phrasings,
192 exposures. Anchor-only and mixed use identical labels. The all-label oracle
uses768 exposures and is not label-budget matched to sparse anchors.

Save full teacher matrices, prompts/token IDs, initial/final adapter and optimizer
states, per-step losses and report probabilities, complete evaluation rows,
source snapshots, counts and hashes. No update while evaluating and no best
checkpoint selection. Report expected true-token probability, full argmax
accuracy, NLL, invalid-output mass, anchored/unanchored and target-label slices.
Compare copying with fixed noise and correction with anchor-only. A/B-only
agreement identities from prior binary experiments do not automatically apply
when invalid actions receive a third feedback type.

No paper greenlight or GPU expansion follows automatically. Success would be
evidence worth evaluating for a broader study, not a guarantee of novelty or
acceptance. A failed clean stage is a teacher-learning limitation in this setup,
not a refutation of the underlying Hindsight hypothesis.

## Verified result — September 4

Frozen runner commit: `613038a`. Evidence:
`artifacts/hindsight_full_feedback_cpu_20260904_v1`.
Read-only receipts are adjacent, ending `_verified.json` and `_verified_v2.json`;
the second adds teacher and format summaries. Neither changes evidence.

**STOP_TRUTHFUL_HINDSIGHT_LEARNING_UNQUALIFIED.** The truthful arm completed all
96 updates. No copying, fixed-noise, mixed-anchor, anchor-only, or direct-label
replication arm ran. Elapsed595.35seconds CPU;352 forward batches,1,152 forward
examples,96 backwards and96 updates. No GPU used.

| Measurement | Initial | Truthful hindsight final |
|---|---:|---:|
| Training full-vocabulary accuracy |20/64|43/64|
| Held-out phrasing accuracy |8/32|24/32|
| Held-out true-token probability, mean |.2363|.6503|
| Held-out NLL |4.7666|.6419|
| Minimum held-out A/B mass |.0009185|.1509|

The final model improves substantially, but24/32 is below29/32 required by the
90% criterion. Icon and pet-name domains achieve2/4 each, below3/4 required.
Thirteen held-out rows have A/B mass below.95; two have a non-A/B full-vocabulary
argmax. On training,18 rows have a non-A/B argmax. These are first-token metrics,
not complete generated-response evaluations or independent-user generalization.

The saved truthful teacher has44/64 correct training argmaxes, mean true-token
probability.6836 and mean A/B mass.7711, despite explicit true-preference feedback.
This shows imperfect targets are a substantial bottleneck. Student43/64 versus
teacher44/64 does not prove optimization is perfect, nor establish the teacher
as the sole cause. In the separate acquisition calibration, direct exact-label
supervision reached64/64 training and32/32 held-out phrasings under the same
schedule and initial weights. The comparison shows the learning apparatus can
acquire these preferences; it does not make the hindsight teacher reliable.

Verification checks all19 manifest hashes, source/data/schedule identities,
teacher normalization (maximum absolute log-normalizer error2.89e-8), saved
evaluation arithmetic, report laws, adapter finiteness/change, optimizer moments
and96-step state, and execution counts. Adapter L2 change is4.1628. It is **not**
an independent neural-forward, backward or optimizer-update replay. Fifteen
model-free tests pass. Original evidence, initial/final adapter and optimizer
are preserved. No threshold was lowered and no additional steps were appended.

PI interpretation: there is a positive learning signal, but this run does not
test the central copying-feedback claim because its clean control did not
qualify. Do not label Hindsight disproved. Before another comparison, qualify
a teacher on truthful feedback and output format using separate developmental
examples, then freeze and test on fresh examples. A stronger teacher is a
candidate, not an assumed solution; do not quietly replace the frozen teacher
in this result or present any later run as an unchanged protocol.

The planned sparse anchors cover two independently assigned domain preferences;
there is no designed latent structure connecting all eight domains. Consequently,
absence of improvement on unanchored domains would not alone demonstrate a
failed identification method. Any broader correction claim needs an explicitly
identifiable population model and a fair same-anchor-budget baseline.

Reproduce the artifact audit without model execution (output must be fresh and
outside the evidence root):

```powershell
$env:PYTHONPATH='src;.'
python scripts/verify_hindsight_full_feedback.py --root artifacts/hindsight_full_feedback_cpu_20260904_v1 --out artifacts/hindsight_full_feedback_new_audit.json
python -m pytest tests/test_hindsight_full_feedback.py tests/test_verify_hindsight_full_feedback.py -q
```
