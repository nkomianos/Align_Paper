# New proposal audit and runtime policy — 7 September 2026

**Historical execution state:** several screens below have since completed.
Read [current seven-idea status](USER_SEVEN_IDEAS_EXECUTION_STATUS_20260910.md)
before any launch. Preserve the estimate-based admission policy; do not restart
completed or scientifically stopped jobs from this snapshot.

The user explicitly replaced mid-run wall-clock caps with estimated-runtime budget
planning. Current experiment launchers now pass no timeout to the supervisor.
Normal completion and scientific early exits remain; an operator can still cancel
a broken process. Preflight import checks may time out, but running experiments
and saved-result verification are not killed for passing a wall-clock estimate.

Before each job: remaining hours = 50 minus prior rental hours minus elapsed time
since the current allocation began. Admit only if remaining hours cover 1.5 times
the estimated runtime plus a 0.5-hour reserve. These are conservative admission
margins, not a guarantee against overruns. After completion use actual elapsed
time for the next job. Processes do not stop provider billing. Never rerun a
completed pilot merely because it remains in a historical queue.

## Independent triage of the seven pasted ideas

The pasted budgets assume up to 300 H100 hours, not our 50 H200 hours. Their GPU
envelopes cannot be imported as calibrated H200 measurements.

| Proposal | Decision now | Reason and next gate |
|---|---|---|
| Tabular predictor drift | Implemented; first optional new queue entry | Inference-only and a falsifiable decision-level endpoint. Must improve acquisition beyond random/uncertainty; the decomposition itself is standard. |
| Censoring-aware reasoning | Conditional reserve; no RL queued | Statistical estimators are standard, and the closest truncation-specific manuscript remains unresolved. Need complete-rollout bank and matched-cost gradient variance before RL. |
| Endpoint-sensitive generative acceleration | Defer | ERTACache already addresses error propagation and timestep adjustment. A useful directional estimator must earn its overhead and beat strong cache controllers. |
| Portable process values | Strongest LLM reserve; no rollout bank yet | Real within-problem rank reversals must survive binomial noise, format controls, and feature-conditioned calibrators before adaptation. Multi-policy rollouts are appreciably more expensive than tabular lookahead. |
| Observation-aware video | Reserve | TRACE already measures decoded evidence coverage. The remaining claim needs a controller beating grid plus abstention at matched decisiveness and cost, plus audited video data. |
| Executable action marginalization | Defer | Exact equivalence is attractive, but canonical output formatting is a potentially decisive cheap baseline. No evidence yet that alias marginalization beats it per compute. |
| Fixed-marginal coupling audit | Defer | The nonidentifiability is classical; need a real application making paired causal claims and admissible transformations respecting its structural assumptions. A simulator toy alone is insufficient. |

This is prioritization, not a claim that the deferred hypotheses are false.
No new GPU training campaign, video-data claim, or process-value result is implied.

## Literature findings and novelty limits

- [Prediction-Oriented Bayesian Active Learning, AISTATS 2023](https://proceedings.mlr.press/v206/bickfordsmith23a.html)
  already proposes EPIG in prediction space. Our information score is not a new
  information-theoretic objective simply because it is applied to a TFM.
- [PFN martingale posteriors](https://arxiv.org/html/2505.11325) explicitly identify
  martingale violations and use PFN predictions to initialize coherent subsequent
  nonparametric updates. Pairwise matrix scaling here does not make the full
  predictor globally Bayesian and does not replace that baseline.
- [Active In-Context Learning for Tabular Foundation Models](https://arxiv.org/html/2603.27385)
  implements uncertainty/diversity methods and finds dataset-dependent random
  selection performance. It does not establish predictor drift as the explanation.
  We have not established that raw lookahead KL is a widely deployed TFM rule;
  claiming a general existing-practice failure would currently be a straw man.
- [Do BNNs Actually Behave Like Bayesian Models?, ICML 2025](https://proceedings.mlr.press/v267/pituk25a.html)
  occupies broad audits of Bayesian coherence. A viable paper needs the specific
  downstream selection consequence and a useful correction.
- [Uncertainty Decomposition for Bayes-Filtered Transformers](https://arxiv.org/html/2602.04596)
  (July 2026 revision of the February TabPFN paper) derives predictive CLTs under
  drift conditions and separates uncertainty using predictive-update volatility.
  This further rules out advertising a generic drift/uncertainty decomposition as
  our contribution. Its uncertainty estimator is an additional follow-up baseline.
- [Distributional PRMs](https://arxiv.org/html/2605.06785) use PRM-hidden-state
  conditioned optimal transport and explicitly discuss ranking limitations.
  Comparing a portable method only with scalar monotone calibration would be weak.
- [TRACE / VES-Bench](https://arxiv.org/abs/2608.22516) already audits whether decoded
  frames cover necessary evidence and offers a training-free acquisition agent.
- [ERTACache](https://arxiv.org/abs/2508.21091) is relevant prior work on error
  rectification and timestep adjustment; generic error-aware acceleration is occupied.

These findings narrow the proposals; they do not constitute an exhaustive novelty
clearance. Full-text truncation and action-redundancy comparisons remain prerequisites
for moving their projects out of reserve.

## Implemented tabular screen

Five development datasets: iris, wine, breast cancer, synthetic linear classification,
and synthetic moons. Two initial-pool seeds per dataset, nested within dataset.
Initial context has 12 labeled examples including two per class. Unlabeled target
pool has 16 rows; separate evaluation has 32 rows. Four fixed rounds each expose
the same fresh eight candidates to every arm. Candidate blocks are disjoint across
rounds; this deliberately restricted acquisition setting is not unrestricted pool AL.
Outer standardization is fitted on initial-context features once. The TFM's own
context-dependent preprocessing remains part of its update and needs later controls.

Five arms: raw expected KL, drift-subtracted mutual information, fixed-marginal
KL projection using iterative matrix scaling, predictive entropy, and seeded random
selection. All lookahead arms use identical candidate/target pools. Hypothetical
labels are enumerated. True candidate labels enter context only after selection;
evaluation labels never enter acquisition. Probabilities have a fixed 1e-12 floor.
Projected information is a secondary variant, not selected after seeing outcomes.

Primary screen: mean evaluation negative log likelihood over acquisition rounds.
The prespecified information arm must improve over raw, entropy and random by at
least 2% and change at least 20% of initial raw-vs-information selections to emit
DEV_SIGNAL_REQUIRES_SECOND_FAMILY. Otherwise stop this version for lack of decisive
development advantage. This gate selects projects; it is not statistical evidence
of a population effect or the proposed 10% label-efficiency confirmation endpoint.
Record per-dataset effects, calibration, effect uncertainty and wall-clock costs
before expanding. Dataset is the eventual replication unit, not 4 rounds or 8 rows.

Logistic-regression runs and single-case smoke runs always emit
DEVELOPMENTAL_PIPELINE_ONLY. No output can green-light a paper. CPU TabICL execution
uses the actual model and is not simulated GPU output. A full CPU development run
has been started to avoid spending GPU time on a workload that may finish locally.

The official TabICLv2 checkpoint (110,368,038 bytes) is downloaded locally and
SHA256-pinned to `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
Official source is pinned to `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`.
Its fit method normally reloads weights each call. Our subclass reuses immutable
loaded weights while rebuilding context preprocessing; KV caching is disabled.
Four ensemble estimators use a fixed seed. The repeat/reset control passes on CPU:
after an intervening fit with different labels, original predictions are identical,
agree with a freshly loaded model, and all weights remain unchanged. Another
predictor family is still needed before a broader TFM claim.

Raw probabilities, contexts, shortlists, selections, checkpoint/source hashes,
and runtimes are saved. The verifier recomputes decomposition, projection,
selection, context history, NLL and routing. It does not rerun neural weights.

## Runtime estimates and launch

Initial TabICLv2 CPU smoke: 86 prediction calls in about 32 seconds on iris.
This is not a whole-suite or H200 benchmark. GPU planning estimate for the fixed
tabular screen is 2 hours conservatively, replaced with host measurements before
admission when available. Existing estimates: Hindsight 1h, compensation 3h,
reference 2h, monitoring 2h, CLARA CPU 0.05h. With tabular and all optional jobs,
the sum is 10.05h; without monitoring/Hindsight it is 7.05h. Avoid duplicate GPU
tabular execution if the local CPU run already supplies the planned evidence.

Extract the pinned TabICL source ZIP and add its `src` directory to PYTHONPATH
alongside repository `src`, root, and `scripts`. The source remains separate from
our repository and its license remains in its source ZIP. A compatible CUDA PyTorch,
numpy/scipy/scikit-learn/einops environment is required; current preflight checks
the optional import before model work.

Four Linux overlays (x86_64/aarch64, CPython 3.10/3.12) now include scikit-learn
1.7.2, scipy 1.15.3, einops 0.8.2, joblib 1.5.3 and threadpoolctl 3.6.0, with
dependency closure checked alongside the existing application wheels. CPU evidence
used local numpy 2.4.2, sklearn 1.8.0 and scipy 1.17.1; these version differences
must be recorded, not silently described as bitwise-identical environments.

```bash
python scripts/launch_research_suite.py --out "$SUITE" --snapshot "$QWEN_SNAPSHOT" --tabular-checkpoint "$TABICL_CHECKPOINT" --tabular-inputs "$TABULAR_INPUTS" --allocation-start-utc "$ALLOCATION_START_UTC" --previous-h200-hours "$PRIOR_H200_HOURS" --dry-run
```

Remove `--dry-run` to execute the admitted queue. Optional `--runtime-estimates`
accepts JSON stage-to-hour overrides from measured workload benchmarks. Omitting
the tabular checkpoint omits that job. Add Hindsight/monitor flags only when their
data are qualified. Scientific early exits are retained; no automatic replication.

Use the new 7 September package, not the 6 September source that still kills runs
at its historical caps. Old packages and results are preserved unchanged.

## Verification at implementation freeze

31 CPU regression cases pass, including the two launchers passing an explicit
`None` timeout, admission arithmetic, exact KL counterexamples, matrix projection,
split leakage rejection, and evaluation-label noninterference. A real POSIX
subprocess with no timeout completes normally. The TabICL CPU smoke and the
full logistic pipeline pass saved-output verification. The logistic control has
no corrected-score win (mean NLL: raw 0.4244, information 0.4247, projected 0.4389,
entropy 0.4240, random 0.4152); this is an apparatus/control result, not a TabICL
hypothesis verdict. Full TabICL development remains in progress at this freeze.
