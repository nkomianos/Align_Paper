# Agent monitoring: statistical and public-data feasibility audit

Status: exploratory local work completed; **no paper green light, no GPU run**.
This is not another trained-agent experiment and is not counted as one.
All previous failed/invalid assays and checkpoints remain preserved.

## Motivation and novelty screen

Two recent primary papers motivate a useful question, but do not themselves
constitute a contradiction: [Last Step Matters](https://arxiv.org/html/2608.29685v1)
finds weak intermediate uncertainty signals on deep research; [Doomed from the
Start](https://arxiv.org/html/2607.06503v2) uses hidden-state probes and calibrated
abort cascades in other environments. The latter explicitly groups train/test
splits by task. Different tasks, features and evaluation clocks preclude
inferring that either paper is wrong. No released trace link for these two
specific experiments was found in the inspected pages; absence from that search
is not proof that no release exists.

Candidate questions considered:

1. Does a monitor identify mistakes within a task, or mostly task difficulty?
   Needs several runs per task under the same frozen policy, paired within-task
   success/failure ranking, task-only controls and heldout task groups.
2. Can retrospective alignment to the final step/last path switch support an
   online intervention claim? Such events are generally unavailable in advance.
   A useful comparison must use observable clocks and explicitly changing
   at-risk populations. This observation alone is not novel or a refutation.
3. Does a new-task reliability certificate remain valid when its nominal sample
   size counts repeated runs of the same tasks? This received the exact audit
   below. The issue is *certification sampling*, separate from train/test leakage.

Prior-work constraints are substantial. [Predicting Task Difficulty Without
Rollouts](https://arxiv.org/abs/2608.05797) already addresses difficulty signals.
[Verify What Matters](https://openreview.net/pdf?id=nv1jzr0FaZ) already distinguishes
uncertainty from downstream intervention value. [Distribution-free inference
with hierarchical data](https://arxiv.org/abs/2306.06342) develops inference for
grouped/repeated observations. [Generalized HCP](https://arxiv.org/abs/2608.15500)
extends prediction after observing part of a new group. [HG-CRC for language
models](https://arxiv.org/abs/2607.24562) addresses risk across a specified group
hierarchy. Do not present generic grouped calibration or an early-abort probe
as our new method.

## Exact certificate counterexample — analytical, not agent measurements

Population: every original agent episode succeeds. A *fixed* abort gate retains
every episode on 95% of tasks and aborts every episode on the remaining 5%.
Tasks are independently sampled. Conditional on a task, rollout outcomes are
deterministic, which also satisfies conditional iid sampling. Actual retained
success recall on a newly sampled task is therefore 95%.

Sample 20 tasks with eight runs each. If all 20 tasks are retained, an
incorrect pooled-binomial calculation treats the record as 160 independent
retained successes. Its 95% one-sided Clopper–Pearson lower bound is

`0.05^(1/160) = 0.9814508659`.

That exceeds a desired 97% recall target. But this false certificate occurs
with probability `0.95^20 = 0.3584859224`, not at most 5%. Counting independent
tasks gives `0.05^(1/20) = 0.8608916593`, and does not certify this record.
In a genuinely iid 160-episode positive control with the same actual recall,
the false-certificate probability is only 0.0025696636.

We exactly enumerate the binomial mass for 12 combinations of 20/50/100 tasks
and 1/4/8/32 repetitions. No Monte Carlo, generated user behavior, or LLM output
is used. This proves a failure of **pooled inference under clustered sampling**;
it does not invalidate Clopper–Pearson under its actual assumptions. Nor does
it establish that the optional independent certification sample in a particular
paper uses this bad sampling scheme. Fresh iid task-episode certification is
not affected by this counterexample. Fixed-benchmark conditional performance
is a different estimand from new-task population performance.

### A valid but standard baseline for success-conditioned recall

For independent task clusters with a fixed number of repeats, let `Y_ij` be
original success and `A_ij` indicate that a gate fixed before certification
would allow completion. Define

`D_i = mean_j [Y_ij A_ij - rho Y_ij]`.

The target is `R = E[Y A]/E[Y]`, not the unweighted average of task-conditional
recalls. For `E[Y]>0`, `R >= rho` iff `E[D_i]>=0`.
Each `D_i` lies in `[-rho,1-rho]`, an interval of width one. Standard one-sided
Hoeffding gives lower bound

`mean(D) - sqrt(log(1/alpha)/(2*n_tasks))`.

Certifying only when this is nonnegative has the desired error control under
independent identically distributed task clusters and a frozen gate. Within-task
dependence can be arbitrary. This bound is conservative, is **not a new method**,
and is not claimed to solve adaptive sampling or post-selection calibration.
Repeated identical observations cannot strengthen it. Zero observed successes
cannot produce a certificate. Unit tests check these properties and the ratio
estimand. A useful paper would need a sharper useful procedure or new substantive
empirical finding beyond this known application.

## Public evidence inspected and actually downloaded

All downloads are pinned and hashed, without authentication or paid inference.
No benchmark hidden test labels were opened.

### Replay Gap

[Public dataset](https://huggingface.co/datasets/ashritha0907/replay-gap-trajectories),
revision `3f3e9f544819afc7fe7faf4a4f5955554e4a15db`.
Downloaded the compact index and card, not full message bodies. Our inventory
counts 896 rows, 56 distinct tasks and six successful rows belonging to only
two tasks. Rows mix models, prompt regimes and forks; they are not iid repeats
of one fixed policy. This release is unsuitable for a high-recall certification
or robust within-task failure-prediction study. Its original action-divergence
contribution is a different question; our suitability decision does not judge it.

### AgentHazard

[Public dataset](https://huggingface.co/datasets/Anonymousblind/agent-failure-dynamics),
revision `4e448b80d8fd7dec19dacd43312811ee0013f629`.
Downloaded schema, protocol and train split only. The actual training file
contains 3,043 trajectories, 1,135 resolved, with trajectory ID, scaffold,
edit count, edit outcomes and resolution. It does not expose the underlying
task IDs needed for this cluster audit. Do not infer task groups from edit
sequences, fabricate task identifiers, or treat a scaffold as a task.

### AgentUQ v1.1 — usable offline foundation, different scope

[Public dataset](https://huggingface.co/datasets/changdae/tau2-uq-artifacts),
revision `824d9ec3b53067cb65153fed7c6bbc3815f7e2bb`.
Downloaded all six full trajectory JSONs plus the card, approximately 52 MB.
There are **556 trajectories**: 278 for each of GPT-4.1 and Kimi-K2.5 across
airline, retail and telecom. Exactly one trial per task/model: these data cannot
measure within-policy repeat dependence. No token sidecars downloaded yet.

Recomputed terminal AUROC from released per-role summaries, using failure as
positive class and never choosing a sign after seeing results. Also computed
simple prefix features from fixed counts of generated assistant actions,
excluding the ungenerated greeting. No fitted predictor or threshold selection.

| Agent/domain | Failures / total | Terminal agent NLL AUROC | Final action count AUROC | First-two-action mean token count AUROC |
|---|---:|---:|---:|---:|
| GPT / airline | 29 / 50 | .361 | .663 | .580 |
| GPT / retail | 56 / 114 | .597 | .435 | .424 |
| GPT / telecom | 55 / 114 | .624 | .639 | .518 |
| Kimi / airline | 20 / 50 | .442 | .677 | .538 |
| Kimi / retail | 63 / 114 | .478 | .291 | .546 |
| Kimi / telecom | 4 / 114 | .636 | .651 | .578 |

These are exploratory descriptive values, not validated deployable predictors.
Four failures in Kimi telecom are inadequate for strong generalization claims.
Final action count/terminal NLL are unavailable before termination. Later fixed
prefix cohorts exclude runs that already ended; their AUROCs cannot be read as
pure within-population learning curves. Length effects even change direction
between domains. No general early-warning improvement established.

The release already documents role-tag corrections and GPT tool-argument
logprob absence. Our counts reproduce GPT text/tool call availability (some
tool+text messages carry text logprobs, not argument logprobs); Kimi covers
both types but includes reasoning/markup. **Do not advertise the authors'
already-corrected data issues as a new discovery.**

## Artifacts, verification, decision

- `artifacts/agent_certification_data_audit_v1`: first two public data slices.
- `artifacts/cluster_certificate_audit_v1`: exact cases and slice inventory.
- `artifacts/cluster_certificate_audit_v1_verified.json`: read-only replay.
- `artifacts/agentuq_public_v11`: six untouched public trajectory files/card.
- `artifacts/agentuq_inventory_v1`: per-trajectory descriptive records and scores.
- `artifacts/agentuq_inventory_v1_verified.json`: read-only replay.
- Source: `cluster_certificate_audit.py`, `agentuq_inventory.py` under
  `src/interaction_sprint`; `scripts/verify_agent_monitor_audits.py`.

Both manifests and all input hashes verify; deterministic calculation replay
passes. An initial AgentUQ verifier comparison rejected integer-vs-string
histogram keys after JSON serialization. Fixed the verifier normalization and
added an artifact replay/tamper regression test; no evidence bytes were changed.
Eight tests pass. This is not an independent implementation replication.

PI decision: **no expensive expansion and no new-paper claim yet**. The cluster
failure is clear but known; initial public slices cannot support the desired
within-task analysis. AgentUQ now gives real accessible data for a more focused
measurement study, but one must establish novelty and intervention utility.
Next safe work is offline analysis/design, not re-running synthetic failures
or paying for fresh rollouts before a useful hypothesis has survived review.
No active GPU experiment or scheduled experiment expansion was created.
