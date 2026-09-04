# External UNDO validation: ICF-Bench audit

No GPU calls, derived dataset, or external-validation result was produced.
This inspection does not change the frozen synthetic training/evaluation sets.

## Public source and licensing boundary

Official repository: https://github.com/qianyuli123/ICF-Bench

Inspected main commit: `ac8149951516cf5770d78edaacc3435bb089417c`.
The GitHub repository API returns `license: null`; its recursive tree has no
LICENSE/COPYING file. The README offers running instructions but does not name
a data or code license. A public repository is not by itself an explicit
redistribution license. Under the task's condition "prepare a dataset adapter
and frozen selection if public license available," preparing a derived release
is on hold. This is an operational provenance check, not legal advice. Root can
seek explicit dataset terms/author permission or choose a licensed benchmark.

## Relevant task and why it is not a drop-in score

`subtask_revision/subtask_revision.json` contains355 examples. It is relevant:
the final request changes one subtask while retaining common subtasks. However
it already includes model answers, judge outputs and success flags alongside
old/new instructions and conversations. A future adapter must whitelist only
the original input fields and separately preserve task rubrics. Never put
published answers/judgments in model prompts or select cases using success
flags. The reference workflow generates free-form answers and uses an LLM
judge; it is not compatible with the current one-token A/B/C/D evaluation.

There is also an annotation/estimand issue. In the first source example, the
old subtask extracts numerical data/dates and the new subtask performs named
entity recognition. Date mentions can legitimately satisfy the new task too.
The stored judge counts them as continued old-task compliance. That observation
alone cannot establish inappropriate semantic residue. This is one inspected
example, not a prevalence estimate or dismissal of the published benchmark.

## Conditional adapter protocol, not implemented or launched

After explicit license confirmation, hash the pinned raw source and choose a
fixed32-example subset by source ID hash, never by published model success.
Strip all output/judge/success fields before downstream access. Independently
annotate whether old and new subtask outputs overlap and whether the final
instruction actually prohibits the old behavior, without model-run results.
Keep all sampled cases in the provenance record; report any eligibility rule
and attrition rather than silently dropping inconvenient examples.

Evaluate base and all frozen adapters using identical generation settings on
the original multi-turn request and the clean new-instruction condition. The
paired estimand is not merely "mentions something old": measure retained
common-subtask competence, new-subtask competence, and specifically prohibited
old behavior. Double-blind human annotation or a separately validated judge is
necessary. Source examples with legitimate overlapping output must not be
counted as failures solely for that overlap. No test-time canonical state
oracle derived from hidden labels may be presented as an available agent tool.

Even a positive result would be a small external developmental check; it would
not replace the strong baseline comparison or replication required for a
paper. No honest runtime is available before generation lengths and judging
procedure are fixed.
