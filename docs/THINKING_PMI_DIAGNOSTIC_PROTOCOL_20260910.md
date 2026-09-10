# Thinking-position PMI diagnostic

Development only. Twenty-four already exposed MATH questions; one native Qwen3-8B
thinking trace per question, maximum 256 generated tokens. Authenticate original
model weights. Save full raw float16 logits. Select the maximum-entropy position
and a fixed-hash random control among prefix lengths 32..255 inside an explicitly
generated thinking span. Selection uses only the base model; no reward filter.
At least 16 questions must have an eligible position with entropy >=1 nat. Retain
all eligible questions, including those below that threshold. Positions from the
same question are paired observations, not independent task replications.

The first attempt failed before generation: the native chat template does not
prefill an opening thinking tag. Corrected v2 preserves native rendering and
requires an actual generated opener for position eligibility. Failed v1 remains
separate. CPU verification independently reconstructs entropy and selection.

Before teacher comparison, freeze six contexts: question, neither, question plus
reference, reference only, question plus unrelated reference, unrelated reference
only. The unrelated reference comes from the next sorted question ID cyclically;
length and subject are not matched, so its interpretation is limited. This extra
control was added before any teacher-conditioned comparison on these traces.
The base prompt must match the generating prompt exactly. Append identical
generated prefix IDs in every arm. No artificial closing thinking tag is added.

Compute beta=1, c=10 centered-tanh PMI targets and the question-only density-ratio
control. Retain all seven logit vectors per probe including cached generation
base logits. Report TV, KL, and entropy separately for high-entropy and random
positions. If any recomputed-base versus cached-base TV exceeds .05, numerical
comparability fails for the entire assay pending investigation; do not quietly
exclude the discrepant row. This tolerance is a pre-comparison apparatus gate,
not a scientific effect threshold. Inspect actual discrepancies in the result.

Estimate: trace generation 5-12 minutes, comparison 3-8 minutes on AWS including
model load but excluding weight authentication. No wallclock kill. No training
is automatically admitted. Small or large differences alone do not identify useful
reference information, reasoning accuracy, or transferable learning. A publication
would require held-out outcome evidence and existing distillation baselines.

Relevant new novelty collision: [Privileged, but Biased](https://arxiv.org/html/2608.04794v1)
already compares in-context, alternative correct, incorrect and unrelated targets
with a teacher/student log-probability measure. It also inserts a closing thinking
marker when scoring targets inside reasoning; our open-prefix next-token assay
measures a different quantity. Neither methodological difference proves novelty.
Generic reference-bias or exploration-suppression claims are already occupied.

Scripts: prepare_thinking_pmi_traces.py, verify_thinking_pmi_traces.py,
run_thinking_pmi_comparison.py, verify_thinking_pmi_comparison.py. Native span
selection tests and existing target-transform tests must pass before comparison.
