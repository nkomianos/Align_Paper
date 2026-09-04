# Eight-hour GPU allocation audit: Hindsight / report candidate 1

4 September 2026. Decision: **park the current Hindsight apparatus; do not add
another training job merely because GPU time is available**. This is an
allocation decision, not a disproof of preference shaping. No new inference or
training was performed for this audit, and no remote job was launched.

## Evidence inspected

- `HINDSIGHT_PARAMETER_PROBE_V1.md`: real adapter gradients and one-step updates;
  primary own-response/full-KL cosine .999156, and identical heldout 7/8 outcomes.
- `HINDSIGHT_FACTORIAL_PROBE_V1.md`: all released-template non-null
  leave-one-task-out cosines exceed .980. Selecting the few opposite directions
  from the alternate template would not demonstrate a robust mechanism.
- `HINDSIGHT_COMPETENCE_PILOT_V1.md`: 1,250 forwards and 544 backwards, actual
  learned adapters. Qualification 30/32 does not transfer: no-adaptation final
  accuracy 36/64. Full KL improves to 40/64 under copying feedback; projected
  full KL is also 40/64 and slightly worse in NLL. Thus neither uniform harm nor
  useful correction is established.
- `HINDSIGHT_SAMPLING_LAW_AUDIT_20260904.md`: the exact estimator identity is valid
  under its declared sampling laws, but is not an empirical welfare result.

The private numeric labels and hand-designed copying channel do not represent
human preference transitions. A bigger model might repair competence, but that
alone would not create a novel method or validate the original welfare claim.

## Primary-source novelty/feasibility check

[SDPO from user interactions](https://arxiv.org/html/2603.12273v1), sections 3,
4.1 and appendix B, supplies the stopped hindsight signal and claimed gradient
equivalence. Its experiments also discuss weaker hindsight learning signals in
smaller or less instruction-tuned models. The original paper is not evidence
that our synthetic feedback channel has the same learning behavior.

[Privileged Likelihood Is Not Automatically Value](https://arxiv.org/html/2608.09263v1)
already examines own-rollout feedback dependence, cross-fitting and the
distinction between teacher likelihood and useful token credit. The generic
diagnosis is therefore occupied. A sampling identity plus a synthetic sign flip
is insufficient priority evidence for a new ICLR contribution.

[Pass-Rate Weighted Self-Distillation](https://arxiv.org/html/2605.27765v1)
also makes capability/difficulty-aware distillation an existing research area.
Calling a stronger-model rerun a new competence-controlled method would overclaim.

No 1–3 hour training run is recommended for this apparatus. Before reconsidering
it, require: a correction with a distinguishable contribution beyond anchor SFT
or gradient projection; competence on a calibration set drawn from the intended
evaluation distribution; a positive informative-feedback adaptation control;
and predeclared endpoints separating expression from persistent transition.
Calibration cannot reuse the final evaluation for checkpoint selection. A
simulator must not hard-code the desired harm-and-recovery conclusion.

## One better use of the window: short-to-long retraction learning

This is a narrower hypothesis within the existing UNDO direction, not another
unrelated idea to queue: **training on local update/retraction relations may
improve correct terminal decisions on much longer unseen edit histories more
efficiently than terminal-answer training with the same training-token budget.**

Why it is worth a bounded comparison: the existing UNDO follow-up already has a
measured history/control gap (87/96 versus 96/96), whereas Hindsight currently
lacks both the intended harm signature and a useful correction. That gap is
small and task-specific; a ten-point threshold missed by one case does not make
the phenomenon absent. It also does not establish the new learning hypothesis.

The strong predecessor is
[Canonical-Context On-Policy Distillation](https://arxiv.org/abs/2605.30251), which
trains a history-conditioned student against a canonical-context teacher and
reports transfer outside its training task. Generic path invariance and teacher
distillation are not our novelty. The potentially distinct claim must be local
relation training, compositional length transfer and data/compute efficiency.
It needs a direct canonical-context baseline, not merely the original model.

Recommended bounded design for the separate UNDO implementation owner:

1. Same qualified base checkpoint; no adaptation, terminal-answer SFT,
   canonical-context distillation, and local-relation learning.
2. Match training tokens and expose all methods to the same underlying states,
   labels and operation vocabulary; report any extra canonical teacher compute.
3. Train on short histories only, evaluate fresh states at short and long
   depths plus unchanged-state and ordinary state-update controls.
4. Freeze model, examples, wording, training length, optimizer and final-step
   selection before launch. No choosing the most favorable test depth afterward.
5. Judge gain over the strongest learned baseline and preservation of clean
   state accuracy. No algebraic theorem claim from a finite test; no acceptance
   claim from one seed.

This audit does not implement or modify UNDO files, does not authorize a duplicate
job, and is not a claim that the proposed method is novel. The experiment owner
must establish executable readiness and measured runtime. Estimated allocation
ceiling: 1–3 GPU hours including controls; actual feasibility requires a timed
batch. If the separate owner cannot implement a fair baseline in the window,
do not replace it with an easier baseline to obtain a positive result.
