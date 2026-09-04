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
