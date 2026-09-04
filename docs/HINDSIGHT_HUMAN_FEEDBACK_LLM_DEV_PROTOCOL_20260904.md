# Capable-reader Hindsight human-feedback DEV protocol

## Why this follow-up is justified

The frozen classical DEV reader failed: late full-text Spearman was 0.096 and
neither participant replies nor full dialogue improved query-heldout MSE. The
source paper reports materially higher correlations from capable prompted LMs,
so that run rejects the cheap TF-IDF assay rather than establishing absence of
signal in the conversations. This independent follow-up changes the reader class
prospectively and leaves the 20-query confirmation split untouched.

## Frozen model and task

Use text-only `Qwen/Qwen3.5-9B` revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, bf16, SDPA, PyTorch DeltaNet,
greedy decoding, non-thinking native chat template, 8,192-token fail-closed
context ceiling, and 24 generated tokens. No training occurs.

The cohort, query split, third/sixth USER boundaries and four evidence arms are
identical to the classical protocol. The model receives the independently
measured pre-rating and estimates the post-rating. Generated text is immediately
reduced to a strict one-field JSON rating, length, and SHA-256; prompts, raw
completions, participant IDs, and transcript text are not written to evidence.

Before human-data inference, six synthetic statements with explicit terminal
ratings must all parse exactly and land within five points. Failure stops before
any human prompt is run. Every human output must strictly parse; otherwise the
DEV signal cannot qualify.

## Frozen DEV criteria

In addition to complete strict parsing, the same five signal criteria apply:

1. late-full Spearman at least 0.30;
2. late user-only has a positive query-cluster-bootstrap lower 95% MSE-gain bound;
3. late user-only reduces MSE at least 5% versus query-only;
4. late full has a positive lower bound versus assistant-only;
5. late full reduces MSE at least 5% versus assistant-only.

Only a full pass authorizes a separately frozen confirmation runner. It still
does not prove influence, preference shaping, SDPO failure, or paper viability.
The paper-level thesis would additionally require the non-identifiability result,
a faithful learning experiment, and an intervention-based correction.

