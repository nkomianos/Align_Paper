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
