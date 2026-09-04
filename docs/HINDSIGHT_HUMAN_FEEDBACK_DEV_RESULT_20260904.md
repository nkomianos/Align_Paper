# Hindsight human-feedback classical DEV result

## Decision

`DEV_SIGNAL_NOT_QUALIFIED`. Park the fixed TF-IDF/Ridge reader and keep all 20
confirmation query groups unopened. This is an assay-class failure, not evidence
that user replies contain no belief-state signal and not a refutation of the
Hindsight causal-identifiability thesis.

## Verified developmental result

The valid run used 72 records in seven frozen DEV query groups. At the sixth USER
boundary:

| Evidence | Spearman | MAE | MSE |
| --- | ---: | ---: | ---: |
| Query + pre-rating only | 0.2068 | 16.2646 | 418.8530 |
| Assistant only | 0.1038 | 16.5078 | 426.8820 |
| Participant replies only | 0.1186 | 16.5109 | 426.4367 |
| Full dialogue | 0.0960 | 16.4987 | 427.1210 |

Participant replies increased MSE by 7.5837 relative to query-only
(`-1.81%` relative gain; query-bootstrap 95% interval `[-14.5872, 4.7616]`).
Full dialogue increased MSE by 0.2391 relative to assistant-only
(`-0.056%`; interval `[-1.0760, 1.2835]`). All five prospectively fixed signal
criteria failed. Early-prefix comparisons were likewise null or negative.

Evidence is local at
`artifacts/hindsight_human_feedback_dev_cpu_20260904_v2`. Its manifest covers the
aggregate report and de-identified numeric prediction rows. Raw transcript text
and participant identifiers were not written to evidence or model services.

## Invalid first attempt

Preserve `artifacts/hindsight_human_feedback_dev_cpu_20260904_v1`. A bookkeeping
bug reused the final parsed row's target for every record, producing an all-zero
target and meaningless zero losses. No confirmation rows were read. Commit
`188cb51` fixes the row-specific target binding and adds a regression test. The
v1 output must never be pooled with or cited as scientific evidence.

## Next step

The source paper reports materially higher belief-shift correlations from capable
prompted LMs than this query-heldout classical reader achieved. A separately
frozen zero-shot Qwen3.5-9B protocol therefore tests the same user-only and
assistant-only decomposition. It includes an explicit-rating interface check and
stops on DEV unless every fixed criterion passes. Expected GH200 time is under an
hour; there is no reason to reserve a GPU while the instance is unavailable.

