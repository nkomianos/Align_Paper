# EndoPAHF V2: Label-Counterbalanced External Inputs

Status: prospectively frozen before any v2 model output. This supersedes v1 for
future neural endpoints; v1 and its CPU rehearsal remain preserved evidence.

## Motivation

The v1 Qwen3-0.6B interface rehearsal put 19/32 immediate targets in position D
and showed strong target-dependent capability. A model could therefore obtain a
misleading aggregate score from position or option-content bias. The remedy is
specified before running a capable model and does not change the selected base
examples or inspect confirmation behavior.

## Transformation

Each already selected PAHF task is expanded into four cyclic rotations of its
four semantic options. The old and new targets, assistant recommendation,
immediate feedback and delayed probes are relabeled consistently. Consequently,
every base example appears once with its old target at A, B, C and D, and once
with its new target at each position. The underlying product, option texts,
old/new semantic choice and expression/transition world remain unchanged.

Expected counts are 512 learning, 384 development and 1,024 untouched
confirmation variants. Selection and thresholds remain blind to confirmation.

## Required invariants

- exactly four rotations per base record;
- equal old-target and new-target counts over A/B/C/D within every split;
- unique variant IDs and deterministic replay from the sealed v1 manifest;
- identical immediate logs in expression and transition worlds;
- different delayed expression/transition probes for every variant; and
- explicit status as input preparation only, never a paper green light.

Future model reports must show both base-example macro averages and performance
by displayed label/rotation. A capable-model interface preflight must pass each
label, not only the aggregate, before any learning comparison is interpreted.
