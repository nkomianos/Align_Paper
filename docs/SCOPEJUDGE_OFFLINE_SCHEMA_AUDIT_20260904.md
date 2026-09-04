# ScopeJudge offline schema audit

Pinned release `b8b06a65a09e39a4fe1682ef56f48fd14ab74800`, downloaded locally to
`artifacts/scopejudge_release_20260904_v1`. The 10,014,899-byte train file matches
the publisher manifest SHA-256
`63a176fde1933464dc1cbfe3d5adcae8d4991ab0f93368903276c55a74efb527`.
No tool call or embedded instruction was executed.

Observed 100 trajectories, 4,897 tool calls and expert label records, 377 majority
out-of-scope labels, 4,379 call-bearing steps, and 304 multi-call steps. Every label
maps to a recorded call. Observations are stored on their issuing agent step,
so including the whole current step would expose its results before execution.
This is a schema hazard, NOT a claim the original evaluator makes this mistake.

Implemented `scopejudge_views.preexecution_view`, retaining earlier non-system
steps and one proposed call. It excludes the current step's message, sibling
calls, results, future steps, root labels and call extras. This conservative view
is not identical to the paper's five strategies. Earlier observation dictionaries
remain intact and require further nested-field review before model use.

The synthetic exclusion/nonmutation test passes; all 4,897 real views construct
successfully. Serialized JSON sizes total 261,253,751 characters, maximum 381,891
for a single view. These are characters, NOT tokens or cost estimates. Views are
constructed transiently; no duplicated full-history dataset was written.

An initial shell one-liner failed on Windows quote handling before analysis;
the committed script rerun completed successfully. Evidence `view_audit.json`.

Next: inspect original prompt construction, validate nested observation fields,
and group evaluation by trajectory/task family to prevent leakage. The paper
already compares history strategies; prefix construction alone is not novelty.
This audit establishes an available test substrate, not an experiment result or
a demonstrated decision-time information gap.

## Nested schema and label-structure follow-up

All 4,897 observations contain `source_call_id` (string), `content` (string),
and `extra` (null). No nested label leakage was found. The custom view now
whitelists the first two fields and retains prior call IDs to preserve their
association; it does not silently discard result-to-call links. This is schema
hygiene, not a demonstrated defect in the original harness. Both view tests pass,
and all 4,897 real views reconstruct without changing the raw release.
Fresh output: `view_audit_v2.json` (264,022,125 aggregate characters).

The checksum-verified `label_structure_v1.json` reports:

- 73 of 100 trajectories contain a majority-labeled violation.
- 76 positive calls occur in those trajectories' first positive **steps**;
  batches explain why the count exceeds 73. Within-batch execution order is
  not assumed.
- 301 of 377 positive calls occur strictly after that first positive step.
- Of the first-step positives, 25 have 3/5 votes, 18 have 4/5, and 33 have 5/5.
- The release has 31 distinct `task_family` strings. Do not silently equate
  these with the paper's 30 environments; the naming discrepancy is unresolved.

These counts suggest testing *first-boundary detection*, not just pooled call
recall: later violations are numerous, and early labels often have disagreement.
They do NOT establish that any published monitor is worse at first violations,
or that blocking one call prevents all later violations. A stopped/replanned
agent takes a different trajectory, unavailable in this observational corpus.

### Novelty decision and next cheap check

The [original harness](https://github.com/dreadnode/scopejudge) is public and
already excludes system prompts and varies pre-execution history. Its README
also explicitly reports coverage and excludes failed parses from metric
denominators. Neither fact is a newly discovered bug.
[StepShield](https://arxiv.org/abs/2601.22136) already evaluates intervention timing.
Therefore generic earlier-detection metrics are **not a paper greenlight**.

Before any GPU request, inspect the released prediction files and reproduce
their scores. If available, compare first-positive-step and later-positive recall
with trajectory-level uncertainty and vote-stratified sensitivity, using existing
predictions only. This is a descriptive DEV audit. It can reveal whether a
specific failure mechanism deserves a new preregistered test; it cannot itself
justify a new method or a live safety claim. Do not choose a method on these
labels and then reuse them as unseen confirmation.

## Original harness inspection and expert check

Inspected GitHub commit `75bfdb6a36a86919ddbcb89dda47d262e124a1d9`:
`data.py:load_eval_points` snapshots history before appending each current step;
all sibling calls use that same earlier snapshot. `prompts.py` omits platform
system messages. This independently confirms the intended decision-time boundary;
we found no current-result leakage in that construction.
The complete tree has no prediction files. `results/README.md` explicitly says
result JSONL files are not committed. Therefore the proposed reproduction from
public predictions is unavailable in this release (not a claim no files exist
anywhere). No API inference was launched to replace the missing files.

Instead, ran a descriptive leave-one-reviewer-out audit: use >=3/4 *other*
experts as reference positives, exclude their 2/4 ties, and identify the first
positive step separately for each held-out reviewer. This avoids using the
evaluated reviewer's own vote to select its reference label. Results:

| Reference-positive location | Held-out positive judgments | Agreement |
| --- | ---: | ---: |
| First positive step | 283 / 357 | 79.27% |
| Later positive steps | 1,106 / 1,324 | 83.53% |

Difference -4.26 percentage points; 5,000 trajectory-cluster bootstrap resamples
(seed 20260904) give a percentile 95% interval [-11.28, +6.61] points. All five
reviewers remain together in each trajectory resample; they are not independent
replicates. This is pooled positive agreement, **not** accuracy, causal harm,
an expert ceiling, or evidence about model monitors. Reference-positive sets
vary by reviewer. There are 633 excluded tied reviewer-call pairs.

Evidence: `artifacts/scopejudge_release_20260904_v1/expert_boundaries_v1.json`.
Implementation `scripts/audit_scopejudge_expert_boundaries.py`; a synthetic
unanimous/batched test checks grouping and zero difference.

**Decision:** no robust expert first-boundary deficit established, and no released
model predictions to test that separate question. Park this generic first-boundary
direction rather than queue a full inference sweep. It remains a reusable
defensive-evaluation substrate, not a qualified paper candidate. A subsequent
security hypothesis needs a specific new intervention or mechanism, not simply
another first-versus-later plot.
