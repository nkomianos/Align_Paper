# Poison-complexity G0 v4 capability bracket

The final semantic-selector task remains frozen. V3 showed that 0.5B and 1.5B
do not both meet the 85% explicit conditional-selection gate, so those model
sizes are excluded from a poison sample-complexity comparison. That is a valid
capacity exclusion, not a learned-model result.

V4 tests the unchanged task on immutable Qwen2.5-3B-Instruct and
Qwen2.5-7B-Instruct revisions. No threshold, prompt, target, poison count,
training example, or evaluation case changes. Both models must pass the same
90% clean and 85% explicit-rule gates before any weight update.

For a full run, model-specific microbatches of 8 and 2 with accumulation 2 and
8 preserve an effective batch size of 16 and 128 optimizer steps. This keeps
the 7B full-parameter AdamW run within the 98GB device while retaining the
frozen optimization exposure. A finite one-batch runtime preflight must pass
for both models before the factorial launches.

If either larger model fails capability or runtime qualification, Candidate 2
is closed under the available compute budget. No additional task repair or
larger model search is admitted.
