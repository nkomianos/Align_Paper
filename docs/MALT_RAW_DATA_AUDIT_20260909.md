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

## Completed input-link feasibility scan

All 42 default-export shards were hashed and all 741 selected runs scanned.
Across these runs, observed input-node payloads and parent references had zero
conflicts or missing parents, and all input occurrences had node IDs. This is
a useful structural positive, not certification of the executed trajectory.
Twenty-eight runs contain multiple output candidate groups (978 API samples);
128 runs contain unmatched samples (1,406 samples). Every run has at least one
final output signature absent from observed inputs. Terminal outputs naturally
need not appear in another API request: this is not proof of corrupt data or
missing executed behavior. Execution selection still needs source semantics.

Maximum single-request input character counts per run range from 1,255 to
234,021 (median 14,715). Characters are not tokenizer lengths. No truncation or
inference is authorized by this diagnostic alone. The canonical graph export
remains separately gated; accepting malt-public did not grant graph access.

Reports are in artifacts/malt_access_recheck_20260909/linked_input_audit.
Preserved first-pass reports use the misleading key
outputs_not_observed_as_later_input: its calculation is membership across ALL
observed inputs, without a chronological test. The scanner now calls this
outputs_not_observed_as_input, with unchanged arithmetic, and explicitly warns
that matching signatures do not prove execution. Original reports are retained.
Three targeted scanner tests pass, including a repeated earlier-input case.
No monitor GPU run or paper-quality positive follows from this scan.

## Exact task/model overlap and split feasibility

The CPU matching audit finds only eight exact task/model cells containing both
labels: 20 positive and 34 negative runs, supporting at most 12 disjoint matched
pairs. These span six exact tasks in five provisional families. This excludes
27 of the 47 positives from a within-cell comparison; report that selection,
rather than presenting the matched subset as representative of all MALT.
All-pairs rankings reuse runs and are not independent observations. Treat tasks
as clusters and show cell-level results; eight cells are not eight independent
task families. The source-bound report is MATCHING_FEASIBILITY.json.

The family/model incidence graph has one connected component containing all
741 runs. Therefore partitioning ALL these rows into nonempty splits that share
neither family nor model is impossible. Dropping connecting cells can permit a
different partition but changes the population and must be specified in advance.
Model identifiers are narrower than lineages, so grouping true lineages cannot
rescue this all-row split. Separate family-transfer and model-transfer tests are
different estimands and must be reported separately.

Next admissible MALT work: qualify transcript rendering against source semantics,
then freeze a matched developmental diagnostic with all 54 overlap runs before
monitor scores exist. Keep the original family-heldout runner and its minimum
counts unchanged; it cannot be fed this smaller matched cohort as if it satisfied
that contract. A new matched diagnostic needs its own protocol, full-context
baseline and tokenizer/memory qualification. Do not interpret an advantage here
as confirmation or as generalization to new families/models. External data with
broader within-task/model overlap is required for the intended paper claim.

Source-code discovery did not locate a public MALT conversion implementation in
the official report or targeted searches. This is a search limitation, not proof
none exists. The ordered-export card explicitly calls outputs API responses and
allows multiple candidates; it does not supply an execution-selection rule.
The official label definitions describe exploited loopholes and explicit task
violations. Do not silently replace these labels with an unreviewed definition
such as whether any generated candidate mentions a forbidden action.

## Matched raw records and compute qualification

Extracted all 54 overlapping runs into overlap_raw/run_*.json.gz. The manifest
binds every record to its source shard hash and row index, canonical JSON hash,
compressed hash, and matching-selection hash. Extraction roundtrips every nested
value back to the original parquet row; it does not select candidate completions.
First-request inspection across all six tasks finds native chat instructions in
some runs and advisor/file-path/candidate-rating scaffolds in others. Native
chat last-input lengths reach 234,021 characters. No generic last-request parser
has been qualified as a complete executed transcript.

An alternative representation can preserve the entire API audit trail, including
every candidate and its uncertainty, rather than reconstructing an executed path.
That could support classification of source run labels from API records. It
would require a separately stated estimand, reversible deduplication with explicit
request/output-group references, removal of run-level label/model metadata from
model inputs, and qualification on actual token lengths. It would NOT establish
which candidate executed or automatically validate action-grounded evidence.
Do not force these records into the old task/trace ablation schema: task text is
repeated inside scaffold wrappers, so a nominal trace-only prompt can leak tasks.

Unlabeled inference qualification on the pinned Qwen3.5-9B/H100 completed:
4,096/8,192/16,384/32,768 tokens took 1.10/1.10/2.59/4.56 seconds respectively.
Peak allocated memory at 32,768 was 25,216,476,160 bytes; all logits were finite.
These are single forward measurements on repeated benign text, no cache and no
gradients, with last-token logits only. They qualify fit, not comprehension,
real-input throughput, or monitor accuracy. No extrapolation past 32,768 tokens
is certified. Reports are retrieved/lambda_h100_20260909/context_qualification_v1.
