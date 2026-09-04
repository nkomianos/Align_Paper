# Independent v3 apparatus-repair review

The locally verified v2 run remains `UNQUALIFIED_GENERATIVE_APPARATUS`:
explicit calibration joint success16/32, grounding32/32; hindsight joint25/32,
grounding32/32, recovery24/24 among original failed-format cases. Inspection
found duplicate Markdown separator rows for the table condition and equals
signs instead of colons for plain output. These are specific presentation
failures, not evidence against learning from interaction or preference shaping.

One new apparatus repair is defensible. V3 leaves the strict v2 score function
unchanged and adds canonical layout examples plus explicit grammar **only** to
privileged explicit-calibration prompts and corrective future-user feedback.
Examples contain unrelated fixed dummy facts, not the current answer. Ordinary
policy/evaluation prompts remain format-neutral and contain only opaque user
IDs plus fresh grounded facts. V3 uses the same users, fresh case IDs and nonce facts, preserving
all v2 evidence. Success feedback remains unchanged.

This changes the feedback distribution to richer instruction/example feedback;
it does not merely fix an implementation bug. Any result must be described as
that explicitly assisted stationary-format positive control. It is not an
untouched confirmation of v2, evidence about natural human interaction, or a
novel method. Examples may make imitation easier; that is appropriate for an
apparatus control but not evidence of causal preference identification.

The qualification rule and fixed training budget remain frozen before v3
generation. Report overall hindsight quality as well as mismatch recovery:
perfect recovery on original errors can coexist with regressions on previously
successful cases, as it did in v2. No further parser edits, example search,
threshold adjustment, selective case removal, or repeated calibration repair
is authorized by this review. If this attempt is still unqualified, stop this
apparatus route and report its limitation rather than repeatedly moving it
toward a passing score.

The separate verifier is `scripts/verify_sdpo_format_control_v3.py`, pinning
v3 data source and the unchanged v2 checker dependency. It replays from both
archived sources rather than importing mutable current-worktree code. All
loss, token-ID, feedback, budget and checkpoint audits remain those of the
v2 verifier. The earlier v2 evidence was re-verified after adding the shared
loader and retained exactly the same unqualified result.
