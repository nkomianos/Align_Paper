# Protocol events versus agent decisions: no blanket artifact claim

Previous goal turn made progress on observable-prefix monitoring. This turn
completes a separate prediction-target audit, using the same 556 public AgentUQ
traces: 19,929 events, including 6,500 generated assistant decisions. No new
agent execution or GPU work. Fixed greeting excluded; no content, tool arguments,
private reasoning or outcomes are predictor inputs.

## Test

Five task-held-out folds per model/domain. Fit classical unigram and conditional
categorical counts, fixed add-.5 smoothing, unseen-context unigram backoff.
Train-only alphabets; unseen held-out labels become UNK with nonzero probability.
Report task-macro and event-micro log loss/accuracy; exact argmax ties are
randomized uniformly. Save all fold counts, alphabets and per-task metrics.

Four prediction targets/contexts:

1. Every event from its preceding event (includes user messages and tool replies).
2. Generated assistant text/tool-name bundle from the preceding event.
3. Same assistant target from the previous assistant decision, skipping intervening events.
4. Tool-name bundle from the previous assistant decision, conditional on a tool
   call occurring. This does not test whether to call a tool.

Ordered tool-name bundles are categories. No correctness of tool arguments,
task progress or utility is established by predicting their names. This is not
an exact reproduction of any published FSM extraction or LLM-context method.

## Results

Task-macro reduction in held-out log loss, bits per target, versus the matched
unigram baseline. Positive is better. Different target spaces have different
entropy: do not interpret cross-column subtraction as an identified causal
contribution of protocol events.

| Cell | All events | Agent / previous event | Agent / previous agent | Tool / previous agent |
|---|---:|---:|---:|---:|
| GPT airline | 1.002 | .123 | -.071 | .294 |
| GPT retail | 1.109 | .167 | .186 | .418 |
| GPT telecom | 1.375 | .112 | .239 | .488 |
| Kimi airline | .966 | .056 | -.015 | .296 |
| Kimi retail | 1.281 | .262 | .631 | .941 |
| Kimi telecom | 1.401 | .296 | .655 | 1.306 |

All-event task-macro accuracy is approximately 69–77%, with large improvements
over its unigram baseline. When predicting only assistant decisions from the
preceding event, the modal prediction remains unchanged in every cell: accuracy
gain is zero, although log loss improves. Thus top-1 accuracy and probabilistic
prediction again tell different stories.

Keeping the last **agent** decision yields genuine predictive information:
Kimi retail assistant-type accuracy rises 44.78% -> 58.27%, telecom 65.31% ->
75.42%. Conditional tool-type accuracy increases in all six cells; on Kimi
retail 23.98% -> 55.40%, telecom 26.68% -> 70.34%. These are observations of
standard Markov dependence, not a novel learning algorithm or semantic planning.
Airline assistant-history log loss gets slightly worse on both models.

The proposed blanket claim that workflow prediction is *only* protocol plumbing
is not supported. The narrower concern about mixed-event metrics is supported,
but this diagnostic alone is not a strong scientific contribution.

## Prior work and decision

[Automata from Agent Traces](https://arxiv.org/html/2608.23670v1) already explains
that much topology is shaped by the harness and studies higher-order context
and state-conditioned prediction. Our numbers must not be presented as a
refutation of its results. We did not run its model-context or failure-prediction
pipelines. The full-messages versus agent-decisions distinction is a useful
evaluation control, not established novelty.

Do not expand either a generic structural monitor or a blanket artifact-debunk
paper. The classical baselines and public-data tools remain useful. The next
paper candidate needs a materially distinct mechanism or intervention, with
empirical value beyond these baselines, not another change in aggregation.
No GPU run is prepared or automatically executing. Goal remains unmet.

## Evidence

Root `artifacts/agentuq_decision_events_v1`, adjacent `_verified.json` receipt.
Manifest SHA-256:
`5fa18dacae9bc7bbb4776fee10c282a86370d0abd655ff6feba06f83a7210ea7`.
Full source/input hashing and deterministic replay pass (same implementation,
not independent replication). Twenty-six relevant tests pass. New tests cover
held-out count exclusion, unseen labels, context-preserving target filtering,
ignored contents/arguments and a positive control where decision-only prediction
must improve. All earlier evidence remains unchanged.
