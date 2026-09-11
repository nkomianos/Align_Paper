# Memory Graft security S3 backbone-component localization pre-registration

## Question

S2b localizes ordinary trigger learning outside the complete Memory Graft and
into the poisoned Pythia backbone. S3 asks which coarse backbone components are
individually sufficient or necessary for that behavior. It is a checkpoint-only
causal intervention. It does not retrain, select a favorable group, or interpret
parameter norms as storage.

Use all five S1 trainable-table decisive seeds for Pythia-410M and Pythia-1.4B.
Before every intervention, restore the entire graft to its seed-matched clean
state. This holds the already localized boundary fixed. The intact poisoned
endpoint is therefore the S2 `poison backbone + clean graft` condition.

## Fixed intervention families

Two overlapping but prospectively fixed partitions answer different questions.
The functional partition contains input embedding plus output head, all
attention parameters, all MLP parameters, and all normalization parameters. The
depth partition contains all non-graft parameters in decoder layers 0--5, 6--11,
12--17, and 18--23. No group is subdivided, merged, removed, or reordered after
checkpoint values or outcomes are inspected.

For each group and seed:

1. Sufficiency starts from the clean backbone and clean graft, then transplants
   the poisoned checkpoint values for exactly that group.
2. Necessity starts from the poisoned backbone and clean graft, then restores
   the clean checkpoint values for exactly that group.

Report ASR, clean-adjusted sufficiency, poisoned-ASR drop, group parameter count,
and group delta norm. Norms are descriptive only. Because component interactions
can be non-additive, failure of singleton sufficiency or necessity licenses
`no coarse singleton localization`, not a proof of distributed representation.

## Derived thresholds and controls

The downstream security assay was designed to resolve a 0.15 ASR effect. A
component is scientifically meaningful only if it independently accounts for
at least that same effect. Therefore component sufficiency passes when the lower
endpoint of the five-seed 95% Student-t interval for clean-adjusted ASR exceeds
0.15. Component necessity passes separately when the lower endpoint for intact
poisoned ASR minus restored-group ASR exceeds 0.15. There is no AND gate between
the estimands.

Endpoint controls protect interpretation rather than add a scientific conjunct.
Clean-to-clean and poisoned-to-poisoned no-ops must reproduce their source ASR
exactly. Transplanting the complete poisoned non-graft backbone into the clean
checkpoint and restoring the complete backbone to clean must each have a lower
five-seed effect endpoint above 0.15. If these controls fail, the checkpoint
patching procedure cannot express the known S2 contrast and component results
are invalid.

Evaluation uses the unchanged 1,024 held-out contexts and one-token exact-match
score. The training seed is the replication unit. Prompt rows are retained but
are not used as independent replicates.

## Scale and verification

The stage enumerates all eight fixed groups in both causal directions for five
seeds and two model sizes, plus four endpoints. Runner plus full replay is
projected below one GPU-hour, 2.26% of the estimated 44.33 hours remaining. This
scale is justified by complete coverage of the coarse architectural partition,
five independently trained source checkpoints, and the existing 1,024-prompt
measurement resolution. More prompt rows would not increase the number of
independent optimizer outcomes.

The decisive verifier revalidates the complete S1 source manifest, reruns every
endpoint and group intervention from source checkpoints, records raw prediction
disagreement, and requires every registered endpoint and component decision to
reproduce. Configuration, runner, verifier, and this document are hashed before
the first checkpoint is loaded.
