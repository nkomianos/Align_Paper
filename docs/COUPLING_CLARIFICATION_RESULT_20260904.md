# Both-pair clarification: park the native hierarchy

Both frozen runs finished with exit0: 40 new articles,32 new seeds,four policies,
10,240 completions per pair. Small-pair runtime954.00s; larger-pair727.78s.
No output-based changes, omitted questions, or additional seed rounds.

## Verified result

Ratios below are hierarchical F1-difference variance times measured cost,
relative to the indicated baseline. Lower is better.

| Pair | Independent | Token-clock | Byte-clock |
|---|---:|---:|---:|
| Qwen3-0.6B / SmolLM2-360M | 0.968 | 1.098 | 0.971 |
| Qwen3-4B / SmolLM2-1.7B | 0.912 | 0.922 | 0.973 |

Every question-cluster variance-ratio interval crosses1. The predeclared nested
question/paired-seed sensitivity agrees: small-pair intervals are
[0.780,1.115], [0.876,1.234], [0.777,1.139]; larger-pair intervals are
[0.770,1.064], [0.748,1.062], [0.790,1.125], in the same baseline order.
All corresponding covariance-gain intervals cross0. These are descriptive
uncertainty analyses, not evidence that the true effect is exactly zero.

The original eight-question small-model ~34% variance reduction did not
replicate at that magnitude. The larger pair has modest favorable point
estimates, but neither pair establishes a robust practical advantage over
both simple coupled baselines. Do not pool away the small-pair loss or select
only the favorable larger-pair comparisons.

**PI decision: park this heuristic; no further task/seed/model tuning to rescue
this clarification.** This rejects expansion on current evidence, not all
possible cross-tokenizer coupling methods. Exact finite qualification also
rules out retokenization-invariance/optimality claims. A new method would need
a genuinely different mechanism and serious complete-byte comparators.

## Evidence and integrity scope

Local root: `retrieved/coupling_clarification_20260904T0935Z`.
Original raw archive SHA256:
`13900972cbbc0da2fbc87082d7f3929e3f29f5502aa790f3aa6a070ad2ca86b5`.
Portable small bundle SHA:
`45b1a6e3747b85bc5e0b05599f83381fafd117f7e0d198b2778aefb6aa4d3683`.
Portable larger bundle SHA:
`d2f85617899f9fc16f9ccc26ab0cdf3d1a466922f497f2b68d8b9c50dac20e6d`.
Run manifest SHAs:
small `ac8c450c9bbe81bd67c469a4f2f681f21629130ab307bfa2d02b7218619c8e82`;
large `a2133c160d580bb10fd153611700570e0d6866e8c985de92de922f8372a08c0d`.

Verification checked34/35 bundled files respectively, all substantive inputs,
source/tokenizer bytes, complete record coverage, native decoding, clocks,
cache checks, timing consistency and scores. Six large weight files were
independently rehashed remotely, not all locally copied. No neural-forward or
full sampler replay is claimed.

Important qualification: the prepared manifest contains an invalid self-entry.
It is preserved and explicitly disclosed in
[the incident report](COUPLING_PREPARATION_MANIFEST_INCIDENT_20260904.md).
Only that exact pre-run-pinned manifest is admitted by separate post-run
exception tools. Every substantive checksum still passes; an independent
fresh reconstruction reproduces all inputs and the defect byte-for-byte.
The invalid self-entry is **not** reported as a passing checksum. No scientific
scoring code or evidence was changed to obtain these results.

The separate frozen SDPOv3 control started afterward on the same GPU. The
instance is therefore not yet ready for termination.
