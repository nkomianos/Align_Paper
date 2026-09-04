# Matched-task agent selection audit

## Question and data

Can uncertainty choose the successful execution when two agents solve the same
task? This is retrospective cross-model selection, not within-policy failure
prediction or a causal estimate of switching agents after tool execution.

Used all 556 previously downloaded AgentUQ v1.1 trajectories: 278 tasks paired
across GPT-4.1 and Kimi-K2.5. Dataset revision remains
`824d9ec3b53067cb65153fed7c6bbc3815f7e2bb`. The code checks full task definitions,
simulator/environment settings, step/error limits and task coverage before
pairing IDs. The two realized user conversations need not be identical.

Previous user-status turn was no progress toward a candidate: it checked idle
hardware and restated results. This continuation completes two new offline
analyses. No paid GPU resumed and no new LM generation occurred.

## Label-free percentile selection

Fixed orientation (higher means worse), separately ranked per model/domain:
terminal assistant mean NLL, terminal entropy, final generated-assistant-action
count, first-two-action mean completion tokens, first-two-action tool fraction.
No sign tuning from outcomes. Percentiles are transductive, not prospective
calibration. Ties randomize equally; success is expected payoff. All pairs reach
two generated assistant actions.

| Domain | Tasks | GPT-only / Kimi-only successes | Both fail | Best constant | Oracle selection |
|---|---:|---:|---:|---:|---:|
| Airline | 50 | 5 / 14 | 15 | 60.0% | 70.0% |
| Retail | 114 | 25 / 18 | 38 | 50.9% | 66.7% |
| Telecom | 114 | 1 / 52 | 3 | 96.5% | 97.4% |

None of the 15 feature/domain selectors improves over leave-one-task-out constant
model choice. NLL selection scores 47.0%, 50.0%, 79.8%, respectively. This is not
evidence that uncertainty is generally useless: equal percentiles erase model
competence differences. Telecom offers only one additional success beyond Kimi.

## Task-held-out calibration follow-up

Post-hoc exploratory follow-up, not independent confirmation. Five deterministic
folds per domain hold out both executions of each test task. Train-only feature
standardization; standard logistic regression with model identity and
model-specific slopes, C=1, no hyperparameter search. Each of five features
evaluated separately. Baseline chooses a model from success rates estimated on
the same training tasks with Beta(1,1) smoothing. Test outcomes never enter fits.

| Domain | Constant | NLL | Entropy | Final action count | First-two token count |
|---|---:|---:|---:|---:|---:|
| Airline | 60.0% | 54.0% | 54.0% | 56.0% | 60.0% |
| Retail | 50.9% | 49.1% | 48.2% | **57.9%** | 50.0% |
| Telecom | 96.5% | 96.5% | 96.5% | 96.5% | 96.5% |

First-two tool fraction gives 56.0%, 51.8%, 96.5%. Retail action count adds eight
successful selections /114 (+7.02 points), Brier .25285 -> .24027. Telecom NLL
improves Brier .14691 -> .14185 but never changes model choice. Probability
quality and decision value are therefore different measured targets here.

Intervals bootstrap fixed per-task crossfit payoffs, not refitted models or
folds. They do not address multiple feature exploration or benchmark population
sampling. Retail action-count interval [+0.88,+13.16] points is descriptive,
not a confirmatory discovery. Degenerate [0,0] intervals mean identical choices
on these records, not proof of population equivalence.

## Closest prior and missing data

[Progress Advantage](https://arxiv.org/abs/2606.26080) already uses policy/reference
likelihood ratios for agent scoring, selection and failure attribution. Its
[official implementation](https://github.com/deeplearning-wisc/progress-advantage)
is a mandatory stronger baseline; we have not computed its scores here and do
not challenge its reported results. [Trajectory-adapted UQ](https://arxiv.org/abs/2608.11552)
also narrows novelty for generic trajectory-aggregation changes.

The Progress Advantage repository advertises 100 WebShop tasks with eight
rollouts per model. Its configured dataset API returned an authentication/access
error during this check. No files downloaded, no credentials sent. Those counts
are not a verified local inventory. Actual AgentUQ inputs have one trial per
model/task and cannot establish within-policy rollout heterogeneity.

## Artifacts and verification

- `artifacts/agentuq_matched_v1` and adjacent `_verified.json` receipt.
- `artifacts/agentuq_calibrated_v2` and adjacent `_verified.json` receipt.
- Matched manifest SHA: `c2f5d54d377249726743328886ea5e9a3fdc699238f0f2bb7f9aea57d350dfec`.
- Calibrated manifest SHA: `8120c559e9571c6b3411c58f497dee1fee2b6f816f50fa23c54897e238d65270`.
- All task scores, fold IDs, parameters, probabilities and outcomes saved.
- Seventeen relevant tests pass, including held-out-label-change invariance.

Source/data hashes and same-implementation deterministic replay verify. Not an
independent implementation replication. Initial calibrated v1 and receipt are
preserved. Validation was hardened to reject fractional outcomes before integer
casting; v2 gives identical results with updated source hashes. This is one
calibration experiment plus validation hardening, not two scientific findings.

Reproduce without touching source data (output/receipt paths must be fresh):

```powershell
$env:PYTHONPATH='src;.'
python -m interaction_sprint.agentuq_matched --data-root artifacts/agentuq_public_v11 --output artifacts/matched_fresh
python -m interaction_sprint.agentuq_calibrated --input-root artifacts/matched_fresh --output artifacts/calibrated_fresh
python -m interaction_sprint.agentuq_calibrated --input-root artifacts/matched_fresh --output artifacts/calibrated_fresh --verify-receipt artifacts/calibrated_fresh_verified.json
```

## PI interpretation

No paper green light or paid expansion. A mundane terminal-length baseline is
more useful here than tested uncertainty summaries, but it is not a new method.
The remaining practical question is whether observable-prefix progress/error
features add held-out value beyond model identity/length, followed by evidence
of a useful intervention. Do not call terminal selection an early-abort policy
or spend GPU time to reproduce this descriptive finding. Goal remains active.
