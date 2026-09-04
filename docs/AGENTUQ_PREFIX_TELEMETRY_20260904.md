# Observable-prefix telemetry: useful in some cells, not a new paper

Previous goal turn made progress: verified matched-task selection and held-out
calibration. This turn implements and runs an early-prefix baseline on the same
public traces. No GPU, model generation, monitor intervention or new data used.

## Design

Fixed clocks after 2, 4 or 8 generated assistant actions. Exclude the fixed
greeting. Include immediately following assistant-requested tool responses,
but stop before subsequent user/assistant content. Features never access user
simulator reasoning, final rewards, terminal uncertainty, final length or
future-normalized progress. Tool results are matched to observed call IDs;
unmatched responses or absent completion metadata fail validation.

The two-feature baseline uses log mean completion tokens and tool-call count
per action. The eight-feature telemetry model adds distinct-tool fraction,
exact repeated-call fraction (canonicalized arguments), explicit tool-error
fraction, repeated tool-name transition fraction, log mean tool-response
character length and log mean assistant-text length. These are ordinary
observables, not semantic proof of progress. No tool-name vocabulary fitting.

Separate five-fold task-held-out logistic fits per model/domain/clock, C=1,
train-only scaling, no hyperparameter search. Predict final binary failure.
Baseline constant probability uses smoothed training-fold prevalence. Later
clocks condition on reaching that clock, so rows across clocks are not identical
populations. These are exploratory public-data results, not independent trials
or confirmatory tests. No threshold or early-stop action was selected.

## All results: AUROC (length/tool baseline -> telemetry)

| Model / domain | 2 actions | 4 actions | 8 actions | 8-action traces / failures |
|---|---:|---:|---:|---:|
| GPT airline | .389 -> .335 | .550 -> .547 | .625 -> .540 | 30 / 20 |
| GPT retail | .480 -> .449 | .386 -> .391 | .283 -> .322 | 99 / 47 |
| GPT telecom | .530 -> .519 | .475 -> .563 | .573 -> **.700** | 103 / 54 |
| Kimi airline | .379 -> .452 | .457 -> .445 | .236 -> .462 | 27 / 14 |
| Kimi retail | .537 -> .598 | .549 -> .621 | .563 -> .599 | 93 / 46 |
| Kimi telecom | .300 -> .298 | .227 -> .118 | .495 -> .744 | 102 / **3** |

Kimi telecom's final column has only three failures; do not headline its .744.
Telemetry improves Brier over the length/tool baseline in 6/18 cells and over
both length/tool and constant baselines in 5/18. GPT telecom at eight actions
improves Brier .24712 -> .21836 (constant .25267); Kimi retail at four improves
.24897 -> .23530 (constant .24952). There is no consistent cross-setting gain.
No feature attribution or causal role for tool errors is established by these
combined fits. AUROCs pool held-out fold predictions; small cohorts, varying
fold calibration and limited failures require caution. No p-values claimed.

## Novelty review changes the next action

[Automata from Agent Traces](https://arxiv.org/abs/2608.23670) already develops
structural features and prefix monitoring from agent logs. It explicitly
attributes substantial structure to the harness, not only the model.
[Critic Experience Bank](https://arxiv.org/abs/2607.12397) evaluates action-level
confidence informed by prior execution outcomes. Generic structural telemetry
or action-aware confidence is therefore not a defensible new contribution.

More importantly, [Accurate Failure Prediction in Agents Does Not Imply Effective
Failure Prevention](https://arxiv.org/abs/2602.03338) already studies the harm
versus recovery tradeoff and pilot-based intervention decisions. We must not
repackage 'failure prediction is not intervention value' as our new thesis.
None of those methods has been reproduced in this analysis; comparisons above
are only between our explicitly named standard baselines.

The narrower open measurement question worth checking next is which predictive
signal survives when the target is **the next agent decision**, rather than a
harness-generated role/tool-response event. That is only a candidate diagnostic,
not an established novelty claim, ready GPU experiment or paper go decision.
The automata paper's own harness discussion must be credited in any follow-up.

## Verification and PI decision

Root `artifacts/agentuq_prefix_v1`, receipt
`artifacts/agentuq_prefix_v1_verified.json`. Manifest SHA-256:
`fbbb789c27c9bfe407c11f653d7bfb2d54118f624a10ec0e2343d3fd8fa048e5`.
All original inputs unchanged; all prefix feature vectors, train/test IDs,
scalers, fitted coefficients and held-out predictions preserved. Source/input
hashes and deterministic replay pass. This repeats the same implementation,
not independent replication. Twenty-one relevant tests pass, including suffix
and private-simulator-reasoning invariance and held-out-label exclusion.

**Do not expand a generic telemetry-monitor paper.** There is an empirical
baseline worth keeping, but neither consistent improvement, intervention utility
nor method novelty has been demonstrated. No active experiment remains and no
paid GPU work resumed. The paper-finding goal remains active and unfulfilled.
