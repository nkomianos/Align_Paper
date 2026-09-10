# Monitoring policy and transcript recovery: current decision

No paper green light. This is a primary-source novelty check and a developmental
source audit, not a neural monitor result.

[The Guard That Cried Wolf](https://arxiv.org/html/2608.27009v1), released27August2026,
already makes authorization-relative labels central to agent guardrail evaluation.
Its Cautious Bench derives labels from a declared policy, uses matched authorized
and unauthorized cases, separates undecidable cases, and varies object names
while holding authorization fixed. Therefore a generic claim that the same action
can be safe or unsafe depending on authorization, or that policy-derived controls
fix ambiguous guardrail evaluation, is occupied. This does not establish anything
about its performance on our natural long transcripts. A follow-up would need a
specific distinct failure or method, demonstrated against the published controls.

[Adversarial Pragmatics](https://arxiv.org/html/2607.01153v1) also explicitly frames
instruction conflict, embedded commands and policy ambiguity as safety-evaluation
constructs. Its existence further prevents claiming that policy ambiguity itself
is a newly discovered monitor problem. No replication of either paper is claimed.

The CheatBench/source investigation identifies concrete input limitations, but
source defects alone do not establish an ICLR contribution or monitor failure.
Our original observer projection was too permissive and is corrected. Repeated
source trial names are not enough to certify identical episodes. The source
recovery check compares every field, allowing declared reference placeholders
and empty tool arguments as missing evidence; other differences remain flagged.
Eight of20 fuller candidates are compatible under that check. These are separate
new artifacts, not overwrites of released traces or independent safe labels.
The other12 include retained-field differences, some apparently terminal newline
changes. Do not normalize shell text blindly: whitespace can be semantically
meaningful. Compatibility is not proof of execution identity or label correctness.

Artifacts: terminalbench_recovery_v1 under
artifacts/monitor_external_sources_20260909. Reproduction:
scripts/check_monitor_recovery_candidates.py. A new monitor comparison still
requires complete task context, policy/label review and independent task support.
