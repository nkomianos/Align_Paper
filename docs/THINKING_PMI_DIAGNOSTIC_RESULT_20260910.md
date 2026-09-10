# Thinking PMI diagnostic: verified distributions, no learning claim

The corrected cached-schedule experiment completed on AWS. All46probes from23
eligible questions passed independent local raw-logit, context reconstruction,
reference-assignment and metric replay. Cached-generation versus recomputed-base
TV is exactly0 at every probe. Trace generation took395.20seconds; the corrected
comparison took695.29seconds (about11.6minutes; slightly over the3-10minute
estimate). These timers exclude weight authentication and transfers and are not
a provider billing total. No training or complete-answer evaluation was run.

All24 exposed DEV questions were attempted. Twenty-two had a thinking position
with entropy>=1nat;23 had an eligible thinking position. Both selected probes
were retained for all23 eligible questions, including the below-threshold case.
One trace lacked an opening thinking tag and was ineligible by the original rule.
Raw traces (about1.26GB), both comparison runs and the numerical diagnosis were
retrieved locally; their manifests verified. Position selection also replayed
independently on both remote and local CPUs.

| Descriptive mean over23 questions | High-entropy position | Fixed-hash random position |
|---|---:|---:|
| Base entropy, nats |1.452814|0.117398|
| TV: full-reference target vs question-only control |0.229161|0.040434|
| TV: full-reference vs unrelated-reference target |0.208913|0.039992|
| TV: unrelated-reference target vs question-only control |0.168053|0.011299|
| Full-reference target entropy, nats |1.235534|0.125119|
| Question-only control entropy, nats |1.252617|0.118396|

High-entropy median full-reference/control TV is0.173363; random-position median
is2.82e-11. This is a selected-position descriptive contrast, not a population
effect or46 independent task replications. The reference permutation changes
length and subject as well as relevance. Similar mean entropy does not imply
similar target distributions or equal learning outcomes.

What survives: the exact null-reference example still shows that question-only
sharpening can occur without informative reference content. This is a valid
attribution control. On these actual uncertain positions, however, the full
reference target is substantially different from that control. Simple broad
near-equivalence is not supported. Unrelated references also move targets, so
movement itself does not establish useful reference information.

What does not follow: no improved accuracy, transferable learning, harmful
training effect, reproduced thinking collapse, or new remedy is established.
The generic reference-bias thesis also has close prior work. No automatic training
campaign is admitted from this result. A practical paper needs a differentiated
question and held-out outcome measurements, not another plot of logit distances.

Correction ledger: initial tracev1 stopped before generation on an incorrect
native-template assumption. Full-prefix comparisonv1 completed but failed the
predeclared numerical gate at6/46probes, maximumTV0.063341. Base-only diagnosis of
the three largest discrepancies exactly reproduced both original schedules.
Cached comparisonv2 corrected every arm's execution schedule with the same frozen
selection, contexts, target formula and threshold. v1 is retained as an invalid
comparable assay; v2 is a corrected developmental assay, not independent replication.

Artifacts under artifacts/pmi_prefix_diagnostic_20260910:
thinking_traces_v2, THINKING_SELECTION_v2.json,
THINKING_SELECTION_LOCAL_REPLAY_v2.json, thinking_comparison_v1, numerics_v1,
thinking_comparison_v2_cached and THINKING_CACHED_COMPARISON_VERIFIED.json.
Protocol: THINKING_PMI_DIAGNOSTIC_PROTOCOL_20260910.md.

AWS has no remaining compute process from this task after completion. GH200 was
not contacted. Submission decision remains NO-GO; research goal remains active.
