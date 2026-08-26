# Hybrid-Memory G0 Result: Stop

**Frozen source:** `d695c4829b902a011adf245a3dd74d8ca9543663`

**Decision:** `STOP_HYBRID_MEMORY_LINE`

This result is a valid mechanistic feasibility result, not a model-loading,
tokenization, cache-format, or parser failure. The remote G0 v1.1 completed
with exit code zero, and the exact evidence was copied and checksum-verified
locally. Re-running the frozen local analysis produced byte-identical
`report.json` output.

## Pre-registered results

| Quantity | Result | Continuation bar | Outcome |
| --- | ---: | ---: | --- |
| Baseline constraint accuracy | 1.000 | >= 0.90 | pass |
| Baseline authorized-minus-unauthorized margin | 7.311 logits | >= 3.0 | pass |
| Gated-DeltaNet recurrent-state carryover | 0.168 logits | >= 0.50 | fail |
| Bootstrap 95% CI, recurrent carryover | [0.144, 0.193] | lower > 0 | positive but too small |
| Positive recurrent-state rows | 0.875 | >= 0.65 | pass |
| Attention K/V carryover (descriptive control) | 14.449 logits | n/a | dominant storage path |

The causal recurrent-state effect is statistically nonzero, but it is only
about one third of the minimum effect the protocol required before expansion.
The contrastive attention intervention is roughly two orders of magnitude
larger. Thus, after this long context, the model clearly retains the binding
constraint, but the data do not support the central paper claim that
Gated-DeltaNet recurrent memory is a material causal carrier of that safety
constraint.

For an ICLR-scale mechanistic paper, a tiny residual effect in one checkpoint
would be neither a compelling mechanism nor a credible basis for a mitigation.
We therefore do not add template variants, lengths, architecture comparisons,
or a restoration intervention to rescue this result.

## Evidence inventory

- Successful v1.1 evidence: `retrieved/hybrid_memory_g0_v1_1_20260825/`.
- Initial v1.0 runtime failure evidence: `retrieved/hybrid_memory_g0_initial_runtime_failure_20260825/`.
  The first launch produced no predictions and stopped before analysis because
  the evaluator did not bind the paired cache variable. The fix is preserved
  in commit `d695c48`; the original log and run manifest remain archived to
  prevent a silent provenance gap.
