# G4 result: temporal row footprint of table-dependent behavior

**Status: complete and certified by full replay.**

## Registered result

G4 repeated the verified Pythia-410M G3 setting at table learning rate `0.1`
over the same five independent training seeds. Every seed passed the registered
installation apparatus threshold. Whole-table necessity and sufficiency
reproduced the G3 source result. The 72 hash rows addressed over all
trigger-internal token positions were sufficient with mean ASR effect `0.64824`
and preregistered 95% Student-t interval `[0.16611, 1.13038]`; the lower endpoint
passes the registered `0.15` criterion by 0.01611.

The primary specific-necessity estimand did not pass. Restoring all 72 rows
removed mean ASR `0.87832`, but subtracting the largest matched-control drop
reduced the specific mean to `0.66738 [0.09165, 1.24312]`. The final-position
16-row necessity, final-position sufficiency, and incremental earlier-row
necessity estimands also failed their registered lower-bound decisions.

| Estimand | Mean | Registered 95% interval | Decision |
|---|---:|---:|---|
| all-internal specific necessity | 0.66738 | [0.09165, 1.24312] | fail |
| all-internal sufficiency | 0.64824 | [0.16611, 1.13038] | pass |
| incremental earlier-row necessity | 0.39961 | [-0.27981, 1.07903] | fail |
| final-row necessity | 0.47871 | [-0.14493, 1.10235] | fail |
| final-row sufficiency | 0.30059 | [-0.10618, 0.70735] | fail |
| whole-table necessity | 0.87832 | [0.54184, 1.21480] | pass |
| whole-table sufficiency | 0.63633 | [0.16832, 1.10434] | pass |
| outside-table sufficiency | 0.00020 | [-0.00035, 0.00074] | fail |

## What the seed variation shows

The final 16 rows are not a stable item locus. Seeds 26091301 and 26091303
place a necessary and sufficient copy in those rows. Seed 26091302 has zero
final-row necessity and sufficiency, while restoring the 56 earlier-internal
rows removes all ASR. Seed 26091304 also depends on earlier rows, but a benign
marker sharing the first part of the trigger restores 24 of the same rows and
removes 0.95801 ASR. That conservative control is why the registered specific
necessity claim fails. Seed 26091305 installs more weakly (intact ASR 0.39355)
but remains above the prospective apparatus threshold; its all-internal
transplant is more successful than its intact co-adapted endpoint.

## Scientific interpretation

G4 supports a narrow positive statement: under aggressive table optimization,
the small deterministic address footprint across the trigger history can carry
a sufficient copy of the learned behavior. It does not support the stronger
statement that this footprint is a specific deletion boundary. The learned
locus varies across seeds, and rows associated with shared prefixes can also
control the behavior. This makes whole-component addressability and item-level
isolation distinct even when the behavior enters the table.

The result does not rescue nominal final-row deletion, does not establish
tenant-safe deletion, and does not identify which earlier token position is
the unique mechanism. The benign overlap is a real hash/prefix-sharing control,
not a failed assay to remove post hoc.

## Provenance

- source artifact: `artifacts/memory_graft_security_g4_run1`
- source manifest entries: 24
- source manifest SHA-256:
  `818cd99241dbef65dbe71488fbec9ccf9456a9456dde2295f142613fc537bfe3`
- source runner wall time: 393.29 seconds
- full-replay report: `artifacts/memory_graft_security_g4_verification.json`
- full-replay SHA-256:
  `73b3824263ec5a63d229b7276e0c68ad4de6067fcb62cd743fad55a6445bfc0f`
- full-replay status: pass; all eight registered decisions reproduced and all
  ten prediction files had zero prediction-ID disagreements
