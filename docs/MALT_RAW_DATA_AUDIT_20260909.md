# MALT raw-data audit before monitoring

Authenticated access succeeded after the user accepted access. Downloaded the
default data/public-* configuration at immutable revision
6100006c2fd23d81e5afe2b1cc3bf3f3fff48df1 into artifacts/malt_public_6100006.
README declares MIT licensing; LICENSE downloaded with the data. No credential
was stored in this repository or persisted by authentication.

Read all 42 parquet metadata columns: 7179 rows and 7179 unique run IDs.
There are 694 manually reviewed normal runs and 5353 unreviewed normal runs.
The latter cannot be treated as clean negatives. Remaining labels distinguish
prompted behaviors and several natural behaviors; they are not interchangeable.

For the proposed narrow natural constraint-violation estimand, select reviewed
unprompted runs with bypass_constraints or ignores_task_instructions as positive,
and reviewed normal as negative. This yields 47 positive and 694 negative runs
across 25 task-family prefixes. ai_rd_restricted_mlm supplies 20 of 47 positives
but only two negatives; other positive families supply 1–8 each. Task identity
and agent lineage are material confounds. Family is provisionally the prefix
before '/' in task_id; confirm against the source task definitions before
claiming a family-disjoint experiment. Source metadata and cohort counts are
saved in artifacts/malt_access_recheck_20260909.

Inspected raw run 247566 (machine_learning_local/cifar10). Its 71 API samples
include scaffolding advice and candidate-rating requests. The final sample has
multiple output candidates. Concatenating every output would conflate proposals
with executed actions; taking the last API call alone would require correctly
parsing its nested transcript. Therefore a simple JSON-to-monitor conversion is
not yet qualified. Do not mark generated normalization as reviewed merely
because the original run label was reviewed.

Next: inspect the canonical transcript-graph schema or official conversion code,
establish a lossless action/transcript mapping, audit actual task instructions,
then measure complete-token lengths with the pinned monitor tokenizer. Do not
truncate away evidence to satisfy the existing 4096-token runner limit. Freeze
the split and report task-only/trace-only/full-context controls before inference.
The 47 natural positives cannot support broad claims about hidden intent or
reliable detection of arbitrary real-world sandbagging. A small DEV comparison
would remain developmental until independently validated.

Access is now unblocked; normalization, experimental design and capability
qualification are still incomplete. No MALT GPU experiment has been run.

## CPU metadata-only leakage screen

Implemented audit_malt_metadata_baseline.py with fixed logistic regression,
one-hot metadata, five-fold run-stratified versus family-grouped splits, and
no trace text. All preprocessing is fitted inside each training fold. Fixed
cohort: 741 reviewed unprompted runs, 47 positives and 694 negatives.

Pooled out-of-fold AUROC for family+model is .986602 under random run splitting
and .540760 under family-heldout splitting. These numbers alone exaggerate the
interpretive contrast: pooling differently calibrated heldout folds can distort
AUROC. The saved second report includes each fold's positive count and AUROC;
within-fold test-size-weighted AUROC for family+model is .692419 on family holdout.
Model identity alone remains .740254 within folds. Family-only predictions on
unseen families are constant within a fold and cannot discriminate there.

Conclusion: metadata substantially predicts the labels, and family separation
alone does not remove model-lineage confounding. This supports stronger controls,
not a claim that a zero-shot monitor actually exploits metadata or that METR's
published monitoring results are invalid. Their study uses a different population
and prompted LLM monitors; our fitted classifier is not a reproduction. The
[source report](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/)
already warns about sampling bias, label frequency, and limited natural diversity.

Reports: METADATA_BASELINES.json preserves the original pooled analysis;
METADATA_BASELINES_WITH_FOLDS.json adds the corrective within-fold analysis.
Both reside in artifacts/malt_access_recheck_20260909. No p-values, population
prevalence estimates or paper go decision are inferred from these diagnostics.

Further raw inspection: native-chat run 342367 includes node_id, parent_node_id,
branch_id and timestamps on input messages but not on newly generated outputs.
Its final sample is marked unmatched. Other scaffolds wrap prior trajectories in
candidate-rating prompts. A union of observed input-node IDs may support partial
reconstruction, but it is not yet evidence that the full executed terminal path
is present. Keep this distinction when testing against canonical graph data.
