# Verified initial H100 results

This is a developmental results memo, not a submission-ready manuscript.
Remote calibration score/routing verification and supplemental prompt/token/loss
audits passed. All 3328 forward records, 768 gradient microbatches and optimizer
step counts were checked. The 6.8 GB raw calibration archive is retrieved and
its remote/local SHA256 matches; local arithmetic replay is also being completed.

| Final held-out metric | Initial | Supervised | Frozen teacher | Updating teacher |
|---|---:|---:|---:|---:|
| Full-vocabulary NLL | 2.0406 | 1.2495 | 1.4655 | 8.2368 |
| Choice accuracy | .1484 | .3359 | .3203 | .2500 |
| Conditional target probability | .1942 | .2907 | .2573 | .2505 |
| Mean within-base position range | .2735 | .0442 | .2685 | .9995 |
| Acquisition gate | — | Pass | Fail | Fail |

32 base tasks each have four answer-order rotations. They are not 128
independent tasks. One training seed and previously exposed learning population;
no confirmation data opened. SFT clears a practical improvement gate, not a
high-accuracy external-validation bar. Frozen-teacher NLL improves, but its
position dependence fails the fixed .20 bound. Do not describe it as complete
failure to learn or claim that freezing alone solves the problem.

Independent argmax recomputation from raw full-vocabulary final logits yields:
supervised A/B/C/D counts 35/40/35/18; frozen teacher 95/33/0/0; updating teacher
128/0/0/0. Updating-teacher conditional position range .9995 and constant A
prediction show answer-label collapse. All arms have high answer-token mass,
so whole-vocabulary format compliance is not the relevant failure here.
This does not identify latent user welfare or establish a new collapse mechanism.
Existing self-distillation literature already covers related failures.

Compensation original and its single stronger-dose repair both end with verified
STOP_INVALID_EDIT_FORMATION. No recovery training occurs: there is no qualified
perturbation to compensate for. Stop dose/site expansion for this version.
Reference pilot ends with verified STOP_NO_REFERENCE_ADVANTAGE under its frozen
prompted-organism screen; no trained/natural sandbagging conclusion follows.
CLARA repeats the exact CPU baseline result with no neural expansion warranted.

Current thesis status: no candidate is paper-qualified. MALT access is available,
but canonical transcript normalization and task-family controls remain prerequisites
to the external monitoring screen. The graph export is separately gated and the
authenticated request returned 403; access requested from the user. Meanwhile,
finish evidence preservation and numerical/platform fixes rather than repeat
failed screens to occupy the GPU.
