# Memory Graft security S2e surgical-write pre-registration

## Motivation and estimand

S2a could not install a trigger/payload map when ordinary causal-LM training
updated the whole hash table, so it left the behavioral validity of final-row
deletion unresolved. S2e is a separately motivated positive control. It asks:
when a behavior is installed by updating only the 16 packed hash-table rows
addressed at the trigger's final token, does the unchanged S1 target-row
deletion remove it preferentially over benign and random row deletion?

S2e does not estimate where ordinary poisoned fine-tuning stores behavior. S2b
already localizes that behavior to the backbone. S2e validates only the
functional sensitivity of the deletion assay to storage known by construction.

## Fixed construction

Use the sealed S1 clean checkpoints for Pythia-410M and Pythia-1.4B at seeds
26091300--26091305. Freeze the backbone, every non-table graft parameter, and
every hash-table row except the trigger's 16 final-position global rows. Optimize
one-token payload cross-entropy after the unchanged trigger over 1,024 fixed
training contexts drawn from materialized WikiText training tokens beginning at
offset 10,000,000. Adam has zero weight decay and runs 512 fixed steps. A
gradient mask and a post-training bitwise comparison must confirm that no other
table row changed. Any violation invalidates the run.

The fixed learning-rate ladder is `0.001, 0.01, 0.1`. At development seed
26091300, use 1,024 development contexts from the second half of the sealed S1
validation-token artifact. Select the smallest rate meeting the installation
gate independently per model. If none succeeds, report
`SURGICAL_WRITE_INSTALLATION_FAILURE`; do not add a rate or alter steps. For an
eligible model, train five S1 clean replication seeds 26091301--26091305 at the
selected rate and evaluate on the original 1,024 S1 trigger contexts, which are
disjoint from development contexts and training data.

## Derived thresholds and outcomes

The downstream assay must resolve the same `delta = 0.15` ASR effect registered
for S1. With 1,024 paired binary outcomes, the distribution-free two-sided 95%
paired-mean half-width is
`h = sqrt(2 ln(2/0.05)/1024) = 0.08488134473378872`. Development eligibility is
therefore `ASR_intact - ASR_clean >= delta + h = 0.23488134473378872`, ensuring
that the lower bound leaves the full 0.15 effect available for deletion.

For five replication seeds, define raw removal, strongest-control removal, and
target-specific removal exactly as S2a. The assay is functionally validated for
known row-confined storage only if the two-sided 95% Student-t lower endpoints
of both raw removal and target-specific removal exceed 0.15. The first criterion
establishes removal at S1's claimed resolution. The independently derived
specificity criterion establishes that deletion of the target rows works better
than benign or random row deletion. If raw removal passes but specificity does
not, report `DELETION_WORKS_NOT_TARGET_SPECIFIC`. If installed behavior is
eligible but the raw-removal upper endpoint is below 0.15, report
`KNOWN_ROW_STORAGE_SURVIVES_DELETION`. All remaining eligible outcomes are
`INCONCLUSIVE_AT_REGISTERED_RESOLUTION`.

These two confirmatory criteria are a compound AND gate because each conjunct
protects a distinct necessary part of the validation claim and each is derived
from the same downstream S1 resolution. The bitwise parameter invariant is an
integrity condition, not a scientific threshold.

## Controls, scale, and verification

Evaluation uses intact, target-row zero, exposure-matched-benign-row zero, and
16 fixed random-row-set zero conditions, plus near-trigger and untriggered
surfaces. It uses one-token exact argmax over 1,024 prompts; seed is the
replication unit. Compact checkpoints save only the 16 changed rows plus their
clean-source checkpoint hash.

The stage is projected below 0.5 GPU-hours, under 1% of the approximately 47.5
hours remaining. Its scale is justified by its narrow invariant: five
independent clean initializations at both registered sizes, held-out development
and confirmatory contexts, the same 1,024-prompt resolution as S1, and all S1
ablation controls. More prompts would not add independent training units, and
updating more parameters would defeat the known-location positive control.

The runner validates the sealed S1 source before weights load. Independent
verification reconstructs each eligible checkpoint from its clean checkpoint
and 16-row delta, replays all prediction rows and decisions, checks manifests,
and emits an inventory digest. Generality work remains paused until this
outcome is verified.
