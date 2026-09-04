# EndoPAHF Cluster-Aware Confirmation Statistics

Status: prospectively frozen before any capable-model or training endpoint.

## Experimental unit

The four cyclic label rotations are controlled views of one PAHF base task.
They are not four independent users or examples. All primary inference therefore
first averages loss and accuracy across the four rotations for each base task,
then performs paired inference over base tasks. Confirmation has 256 independent
base clusters, not an effective sample size of 1,024.

## Primary comparison

For every base task, compute anchor-baseline old-target normalized log loss minus
augmented-method old-target normalized log loss, after averaging each arm over
the four rotations. The augmented method qualifies only if:

- mean paired log-loss gain is at least `.05`;
- the lower endpoint of a deterministic 10,000-resample paired base-cluster
  bootstrap 95% interval is above zero; and
- mean old-target accuracy is noninferior by a margin of `.02`.

Panel predictions must be averaged prospectively rather than selecting the best
panel using confirmation outcomes. Accuracy, label/rotation breakdowns and
new-target performance remain secondary diagnostics.

## Prospective power audit

Before model endpoints, simulate 400 confirmation studies with 256 base units
under null, `.08` signal with paired-effect SD `.35`, and the same signal with SD
`.50`. The rule qualifies only if null false positives are at most `.06`, signal
power at least `.85`, and noisy-signal power at least `.60`.

This audit validates a decision rule, not a model effect. Confirmation remains
locked until the development gates and external-run protocol authorize it.
