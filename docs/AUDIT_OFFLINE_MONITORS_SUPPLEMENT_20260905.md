# Supplemental offline audit: agent monitoring, ScopeJudge, BeyondMasks

5 September 2026. This report is supplemental to the completed Hindsight data audit. It does not change its counts. The source collectors were inspected before replay; their network/download entry points were not invoked. No model inference, GPU work, host contact, new agent rollout, or locked confirmation access occurred.

**No omitted paper-qualified positive was found.** There are real descriptive measurements here, but they are released-data reanalyses, known statistical calculations, or source/corpus checks. None supplies an independently qualified neural or intervention result. The existing decisions to park generic monitor, first-boundary, and compositor-support paper claims remain justified by the evidence available, without interpreting exploratory nulls as universal falsification.

## Coverage and count convention

The supplemental ledger covers **12 physical artifact roots**, each assigned **Developmental/apparatus only**, plus three grouped **Not run** follow-ups. These root counts must not be added to the main experiment count: two roots contain only downloaded data, calibrated v1/v2 duplicate one analysis, and the ScopeJudge root contains several distinct analyses. All measured endpoints reuse public data; there were **zero new agent/model/editor experiments** in this scope.

The covered roots are `agentuq_public_v11`, `agentuq_inventory_v1`, `agentuq_matched_v1`, `agentuq_calibrated_v1/v2`, `agentuq_prefix_v1`, `agentuq_decision_events_v1`, `agent_certification_data_audit_v1`, `cluster_certificate_audit_v1`, `beyondmasks_contract_audit_20260904_v1`, `beyondmasks_samples_20260904_v1`, and `scopejudge_release_20260904_v1` under `artifacts/`.

Seven committed verifier calls produced **six passes and one failure**. The failure is calibrated v1 against the current source: its source hash differs after documented validation hardening. Independently comparing v1/v2 finds **exactly identical result cells**, including all predictions and fits. This is a preserved superseded artifact, not an additional scientific run or evidence that its numerical result changed. The original v1 source was not reconstructed, so its historical source-bound replay is not claimed.

Three additional committed ScopeJudge analysis programs were replayed into fresh supplemental outputs; **all three matched their saved current-version artifacts exactly**. The BeyondMasks compositor function was separately replayed and matched its saved microtest. These are additional checks, not additional verifier calls. Thirteen inspected current source files exactly match their recorded latest Git blobs; those commits are source snapshots, **not evidence of prospective preregistration**. Full hashes and commit IDs are in `remaining_source_video_checks.json`.

`research_audits/` contains only three previously covered SDPO single-profile JSON files; it contains no agent-monitor output. Agent-monitor source files found inside historical retrieved snapshots are code copies, not separate analyses or runs.

## Corrections and surviving measurements

### AgentUQ

The six raw JSON files contain **556 trajectories**, comprising 278 matched task specifications executed once by each of GPT-4.1 and Kimi-K2.5. Full task definitions and relevant simulator/environment settings match across each model pair. This does not make realized conversations identical, supply within-policy repeated rollouts, or establish intervention outcomes.

I independently recomputed 18 headline AUROCs directly from rewards, released terminal NLL summaries, and generated-action metadata, using pairwise comparisons rather than the implementation's rank statistic. All reproduce. The released NLL summaries themselves were not rebuilt from token sidecars: those sidecars are not present in this scope. Terminal uncertainty and final action count are unavailable for early stopping. Later action clocks condition on a trajectory surviving to that clock; comparisons across clocks are not pure learning curves.

All **15 matched percentile selectors** lose to leave-one-task-out constant model selection. I reconstructed percentile ranks and selection payoffs and checked every stored outcome against the raw trajectory reward. Percentile normalization removes model competence differences; this result does not show uncertainty is universally useless. Both completed traces must already be available, so the measurement is retrospective cross-model selection, not live switching after a prefix.

The calibrated follow-up uses task-held-out standard logistic regression, with both model executions of a task held out together. Source inspection and stored fold checks support the claimed train/test separation and train-only scaling. Independently recomputed selection arithmetic reproduces all 15 cells. The largest useful observation is retail terminal action count: **66/114 selections succeed versus 58/114 for the constant**, a gain of **8/114 = 7.02 percentage points**. It is post-hoc on the same corpus and uses a terminal feature. The fixed-payoff bootstrap does not refit ranks, folds, or classifiers and does not address feature selection or population generalization. Calibrated v1/v2 must be counted once.

For observable-prefix telemetry, all saved labels bind to raw rewards; independently calculated AUROC and Brier scores reproduce for both models in all 18 model/domain/clock cells. Telemetry improves Brier in **6/18** cells over the length/tool baseline, and **5/18** over both that baseline and constant probability. Kimi telecom at eight actions has only **three failures**. Prefix construction stops before subsequent user/assistant content, includes responses to already observed tool requests, and does not read final reward or private simulator reasoning as features. No useful intervention or consistent monitoring gain is established.

The decision-event audit contains **19,929 events and 6,500 generated assistant decisions**, independently recounted from raw messages. All 24 saved task-macro and event-micro aggregation cells reproduce. Its task-exclusive count fits and alphabets are appropriate for the stated exploratory prediction task. It does **not** identify tool-argument correctness, semantic planning, or utility. Different scopes have different targets; subtracting their log losses does not isolate a causal protocol contribution. Last-agent dependence remains, so a blanket claim that workflow prediction is only protocol bookkeeping is unsupported. These descriptive results are worth preserving as controls, not promoting into a new paper candidate.

### Cluster certification and public data support

The Replay Gap index has **896 rows, 56 task IDs, six successful rows in only two tasks**. Models, forks, and prompt regimes are mixed. AgentHazard train has **3,043 trajectories, 1,135 successful**, and no underlying task ID field. No validation/test split was opened. Neither slice supplies the intended independent repeated-execution sample for a fixed policy.

All 12 saved pooled false-certificate probabilities independently reproduce using explicit binomial masses and beta-quantile bounds. For example, 20 independent tasks with eight perfectly dependent repeats yield false certification probability **0.358486** at nominal 0.05 in the constructed case. This is a correct known dependence counterexample, not measured agent behavior. The task-level Hoeffding contrast is a standard fixed-gate IID-cluster bound for a ratio of expectations, not a new certification method. Arbitrary within-task dependence is allowed under its assumptions; adaptive gates, task weighting, and dependent task families are not covered.

### ScopeJudge

The public train file matches the publisher's manifest. Raw labels bind one-to-one to **4,897 calls** in **100 trajectories**, with **377 majority-positive calls**, **76 in first positive steps**, and **301 later**. There are **73** majority-positive trajectories and **11** unanimously clean trajectories. Vote totals and majority labels are internally consistent. The **31 task-family strings** are not independently sampled environments.

The conservative custom pre-execution views exclude current results, sibling calls, future steps, root labels, and arbitrary call extras. Current code also whitelists prior observation fields. All **4,897** views replay to the saved v2 aggregate sizes. This is structural hygiene, not a demonstrated flaw in the original harness or evidence of classifier accuracy. Earlier v1 view/label outputs are superseded checkpoints of these checks, not unseen confirmations; current-version replays are the ones claimed here.

Independent leave-one-reviewer-out arithmetic and a separate vectorized trajectory bootstrap reproduce **283/357** positive agreement at first reference-positive steps versus **1,106/1,324** later: difference **−4.263 percentage points**, 95% interval **[−11.283, +6.611]**. All five reviewers remain together in each trajectory resample; **633** tied reviewer-call pairs are excluded. This neither establishes an expert deficit nor equivalence, and says nothing directly about model monitors. Reference sets differ by held-out reviewer, and trajectories may share task families.

The 27 majority-clean trajectories give a *hypothetical* zero-alarm binomial upper bound of **10.502%** under IID safe-episode assumptions. No monitor was run and no zero-alarm outcome was observed. Treating 4,897 calls as independent safe episodes would be wrong. No model prediction files were found in this local release. The original online harness was not independently reacquired during this explicitly offline supplemental audit; prior claims about its remote source cannot substitute for model predictions.

### BeyondMasks

All **14 metadata/source receipt hashes** match. The listing independently gives **180 complete triples, 544 files, and 2,807,857,340 bytes**. The synthetic compositor check reproduces; it demonstrates an algebraic support contract only.

All **nine local video hashes** match their receipts. I decoded them and independently reproduced their frame counts, dimensions, and all nine sampled-frame pixel summaries. The three size-selected cases have **20, 38, and 68 frames**. For case 171, outside-object-mask MAE is **49.63–49.98/255** and roughly **79.93%** of outside pixels exceed a mean difference of 10. Such differences can include intended global lighting and do not identify effect support or invalidate the pair. Equal frame dimensions/counts do not establish temporal alignment.

The pinned DiffuEraser wrapper requires at least 22 effective frames, so the 20-frame case does not directly qualify without a declared temporal treatment. It changes conditioning, priors, and some temporal masks before a final output composite. A raw conditioning-mask swap is therefore not a pure information-loss intervention. Final composite contrasts with fixed generated pixels are possible, but alone are deterministic diagnostics. A further concrete source finding is `read_priori` calling `os.remove(priori)` at line 142: any future invocation needs a working copy to preserve the immutable prior. This source was read, never executed.

No editor output, qualified effect annotation, or complete factorial experiment exists in these roots. The size-selected clips are development data, not unseen confirmation. The evidence does not revive a compositor-support paper.

## Limits and deliverables

This was an offline evidence audit, not a fresh literature search or reproduction of published agent-monitor algorithms. Existing literature-positioning memos were read but their online sources were not refreshed in this bounded scope. No overlooked empirical positive warrants reversing the main paper choice on that basis. All current analyses lack the prospective qualification and independent novelty/utility evidence needed for a submission claim.

The replay checks share original analysis implementations. Independent coverage additionally includes raw reward binding, AUROC and selection arithmetic, all headline aggregations, exact clustered-counterexample probabilities, ScopeJudge label/reviewer statistics and bootstrap, corpus counts and hashes, and video decoding/pixel summaries. I did not independently refit the logistic coefficients, rebuild terminal NLL from absent sidecars, or rerun a published monitor/editor.

Machine-readable files and reproduction scripts are under `artifacts/independent_audit_20260905/data/`: [claim ledger](../artifacts/independent_audit_20260905/data/remaining_agent_monitor_ledger.json), [independent arithmetic](../artifacts/independent_audit_20260905/data/remaining_independent_arithmetic.json), [verifier calls](../artifacts/independent_audit_20260905/data/remaining_verifier_calls.json), [ScopeJudge replay](../artifacts/independent_audit_20260905/data/remaining_scopejudge_replay_comparison.json), and [source/video checks](../artifacts/independent_audit_20260905/data/remaining_source_video_checks.json). Existing evidence and the completed main ledger were not edited. The consolidated parent index separately records these released-data analyses and construction checks; they do not increase its measured-assay count.
