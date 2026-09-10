# Completed MATH and action-gradient diagnostics

No paper green light. Both raw runs have been retrieved and their read-only
verifiers rerun locally. Archive completed_stage4_math_action_20260910.tar.gz:
SHA256 a4685f58213b43b699740d62f39ce4eb3641332b6f1edc510e824e7705df5fd4.
Local root: artifacts/gh200_research_20260910/retrieved_stage4.

## Harder MATH bank: invalid answer-format qualification

Qwen3-8B completed384 continuations in2326.75 seconds including prefix generation.
All16 DEV questions retained eligible prefixes. Eligible DEV strict-parser accuracy
was.4140625, EOS rate.9114583, and parse coverage.5104167. The frozen.95 parsing
requirement failed. Queue stopped before Qwen3-32B; no paired-policy reversal
result or independent-seed replication exists. Do not call policy dependence false.
Manifest SHA256279894c025b39c96bf8b5d09c68fe751fbaa63991dd51a33f38f05013def8d36.

Inspection of the first six non-parsing raw continuations found ordinary boxed
answers, prose after ####, boxed rational answers, and2048-token truncation.
At least some failures are a delimiter/parser mismatch, not an absent mathematical
answer. This inspection is developmental and does not replace the registered
reward. A repaired parser must separate extraction from correctness, handle wrong
but well-formed rational answers, be validated independently, and be frozen before
any renewed scientific comparison. Do not silently promote this old bank to a
prospectively qualified result. No repeated generation or gate relaxation yet.

## Action gradients: negligible variance reduction in this organism

Eight context/grammar evaluations completed in1.00 seconds after model loading
and hashing, substantially faster than the unbenchmarked5–20 minute envelope.
The four cardinal-direction contexts are apparatus, not independent task families.
Last-layer q/v rank2 LoRA parameter gradients were measured; no optimizer updates
were made. Gradient-mean numerical discrepancy ranged about1.08–1.36%, within the
frozen2% BF16 tolerance; initial action masses passed. Local raw-vector replay
reproduced covariance traces and route STOP_SMALL_VARIANCE_EFFECT.
Median variance reduction was1.0988921e-10 (fraction), far below the20% rule.
Manifest SHA256db2d78fd230b0223f791918f33e53f761a77ea924927e9741823bf817175b628.

Raw scores explain the lack of effect: the alternative JSON key order is between
roughly23 and35 log-probability units below the preferred spelling. Matched action
mass does not match within-action alias mass, intentionally. There is essentially
no sampled spelling randomness to average away. Canonical variance is comparable
to alias variance. This is a developmental negative for this restricted neural
organism, not a learning result, nor a universal rejection of Rao–Blackwellized RL.
No conditional learning pilot is admitted by this result. Artificially forcing
equal alias mass would construct a different organism and cannot rescue this one.

## External tool source preparation remains unrun

Pinned public BFCL source at6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 with raw
download hashes. Source root artifacts/action_tool_source_20260910/bfcl_current_v1.
The repository root license is Apache2.0; do not infer additional data guarantees.
prepare_action_tool_learning.py screens single-function, unambiguous scalar
required arguments, omittable optional arguments, matching function names and
unconstrained integer fields. It does not execute tools or certify every source
label. The first eight selected v1 prompts/labels received direct inspection.

V1 generated +1/+2/+3 distractors, making the true argument always smallest. This
was caught before GPU use; v1 is preserved as a flawed construction. V2 chooses
the true argument's numeric rank independently by saltedID hash, then constructs
four neighboring integers. Of400 source rows,98 are structurally eligible across
88 exact function names; selected64 train and32 DEV rows span58/28 names with no
exact-name overlap. Numeric-rank counts25/20/21/30; output-position counts21/25/25/25.
Same-family tools may still overlap semantically; names are not independent domain
families. Three tests check correct-action uniqueness, alias identity, ambiguous
keys, name mismatches and explicit constraints. No neural test or learning run.

This is a BFCL-derived finite-choice dataset with constructed distractors and an
engineered initial policy. It cannot be advertised as a BFCL leaderboard score,
natural tool execution, or uncontaminated generalization. Independent full-row
label/schema/distractor validation remains required before an external experiment.
