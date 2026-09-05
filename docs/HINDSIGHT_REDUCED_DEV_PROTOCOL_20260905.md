# Prospective reduced EndoPAHF DEV protocol v1

Version: `endo-pahf-reduced-dev-v1-20260905`. Status: **implemented for review; no model run, no empirical result, no submission qualification**. This explicitly supersedes the proposed G0/G1 prerequisite route and fifteen-arm G2 development runner. Their frozen sources and artifacts remain untouched. The complete numerical rule is serialized in `configs/hindsight_pahf_reduced_dev_v1.json`; changes require a new prospective version, never an overwrite motivated by observed outcomes.

## Question and access boundary

The question is whether a standard residual/difference-estimation learner improves prediction of a **constructed delayed preference label** beyond same-information simple learners and readouts. It is an offline next-query target for the observed source cohort. It is neither an observed human persistence measure nor the theorem's endogenous-action objective `P(a=Z1(a))`. The first answer-token reverse-KL implementation is a stated SDPO-style adaptation, not released multi-token SDPO.

Only `MANIFEST.json`, `learning.json` and `development.json` from the pinned v3 input root may be opened. The input manifest SHA256 is `2ae32c119087d97de6f2e5a65959b4c94a7d81362732871ff806b157e000fab2`. Manifest entries naming reserved content are metadata only; the runner/verifier must not open, hash, tokenize or score their targets. No split override exists. The loader validates 630 learning bases/2,520 rotations and 96 DEV bases/384 rotations, exact IDs, user bindings and disjoint learning/DEV surfaces. All four option rotations are grouped within task. Source user means the display name before the first colon in the prompt, since no independent user-ID field is provided; this grouping proxy is not evidence of unseen-user transfer.

All learners may access 630 immediate logs. Every sparse learner and deterministic readout receives the same union of four outcome-blind, disjoint sixteen-base learning-anchor panels, selected using the historical salt `endo-pahf-g2-panels-v2`: **64 unique delayed-labeled bases**. One model per sparse method replaces the old four-panel ensemble. The full-delayed oracle has 630 delayed labels and is an extra-information acquisition control, never a fair comparator.

The runner builds a public immediate-record whitelist and a separate 64-anchor target dictionary. Sparse loss code receives no delayed record/old target outside that dictionary. The full oracle's target dictionary exists only in its control arm. Loading the original corpus for construction validation and the explicit oracle is recorded separately from what each learner consumes. Schedule selection reads IDs/rotation/user structure, not nonanchor delayed labels. Tests remove nonanchor delayed labels entirely to check that schedule/preflight/readout builders still work.

## Exact qualification, deliberately changed before any endpoint

The historical capable-interface panel used DEV. **This version changes panel selection prospectively to sixteen bases from the pooled 64 learning anchors**, using the existing preflight salt `endo-pahf-capable-preflight-v2-exact-hindsight`. This prevents exposing DEV hindsight contexts to the model and keeps preflight delayed labels inside the sparse learners' shared label budget. Four rotations and two contexts produce 128 score-only jobs. This is the same serialization, with a newly disclosed population restriction; it is not a replay of the historical DEV panel.

Serialization is the plain downstream prompt plus the exact existing block:

```text


=== HINDSIGHT CONTEXT ===
[The following is a future user message. Use this to guide your answer to the user prompt.]
{follow_up}
```

The tokenizer applies one user chat message, `add_generation_prompt=True`, `enable_thinking=False`, no second template application, no truncation. All computations are forward logits at the first native answer-token position; `generate` is never called. A/B/C/D must be distinct single native tokens.

Retain each historical context×target-letter cell's 16 examples, at least 14 correct conditional argmaxes, mean conditional target probability at least .70, and mean full-vocabulary answer-set mass at least .10. In addition, this new version checks position at the **paired base×context** level: mean conditional target-probability range across four rotations at most .20, maximum range at most .50, and semantic-argmax disagreement in at most .25 of base×context groups. These last paired limits are new prospective engineering criteria, not empirically calibrated guarantees. No global averaging may cancel opposing per-task order failures.

Qualification uses the exact serialized common initial adapter. Interface failure stops before training and is invalid assay/capability. There is no automatic repair or retry launcher. At most one isolated, documented prospective repair may be considered by a new version; a second failed interface parks this model/interface. No existing threshold is relaxed within a run.

## Initialization, schedules and objective

Qwen/Qwen3.5-9B model and tokenizer revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Require an existing `snapshots/<revision>` directory under that model's local Hugging Face cache. All loads use `local_files_only=True`; all weight shards and configuration/tokenizer bytes are hashed, and effective runtime/configuration/source receipts are saved. The launcher requires exactly one CUDA device and has no CPU model fallback. This document does not authorize GPU access.

Keep the preceding G2 defaults to avoid another optimization search: seed 2026090432; LoRA rank 8, alpha 16, zero initial B and float32 adapters in the same Qwen attention/DeltaNet projections; AdamW learning rate 1e-4, betas (.9,.999), epsilon 1e-8, weight decay zero; constant learning rate; maximum gradient norm 1; 35 updates. Explicit AdamW flags are false for amsgrad, maximize, foreach, capturable, differentiable and fused. A framework-added `decoupled_weight_decay=True` field is recorded and validated as an AdamW-derived setting; unknown settings fail. Base weights use BF16/SDPA, kernels disabled, cache disabled, 1,024-token ceiling, eval mode with gradients enabled only for students. Dropout is disabled. There is no checkpoint selection.

Save the common initial adapter. Every arm reloads an identical canonical tensor hash, resets the same RNG seed, and starts a new AdamW with an empty state. Save each arm's physical initial adapter/optimizer checkpoint, final adapter/optimizer checkpoint, ordered parameter names and canonical/file hashes. Verify every optimizer moment and step counter against exactly 35 completed updates. A failed partial arm preserves its adapter, optimizer, progress and completed-step count; a failed run cannot pass as completed because some arms have finished.

The population schedule retains the existing outcome-blind rotation assignment: hash base IDs using `endo-pahf-g2-global-v2`, assign rotations cyclically in that order, then shuffle the 630 selected rows using the fixed seed. Thirty-five batches of eighteen cover each learning base once. This balances rotations, not necessarily the labels actually presented; actual IDs and exposures are retained.

The pooled anchor schedule hash-orders all 64 shared bases with `endo-pahf-reduced-pooled-order-v1`. The first 32 updates visit two bases each under all four rotations: 256 row presentations covering all 64 bases once. The final three updates revisit six bases selected by `endo-pahf-reduced-pooled-revisit-v1`. Every sparse arm receives the same **280 anchor-row presentations**.

Let `K(s,t)=KL(student || stopgrad(current-adapter hindsight teacher))` over the full vocabulary at the first answer token. The teacher is detached within the step and changes along training; it is not a frozen teacher. The six arms are:

| Arm | Per-update objective |
|---|---|
| `raw_immediate` | Mean immediate-teacher K over the population batch |
| `oracle_delayed` | Same K and schedule, full delayed-expression teacher |
| `pooled_sft` | Full-vocabulary target-token NLL on the shared anchor batch |
| `pooled_sdpo` | Mean delayed-teacher K on the shared anchor batch |
| `residual` | Population mean immediate K + anchor mean(delayed K − immediate K), using the same anchor student logits in both terms |
| `mixture` | .5 population mean immediate K + .5 anchor mean delayed K |

The transition construction has identical immediate/delayed text. CPU validation checks that identity. There is no duplicate transition training arm and no claim that a tensor-zero identity is empirical transition validation.

## Prespecified cheap controls

`baseline` is the common no-update model. Two deterministic readouts use only the same 64 anchor delayed labels and public prompt/options/source-name metadata; they do not read DEV targets or the other 566 delayed labels. They are allowed but do not use immediate logs to fit a new target. No learned hyperparameter or confidence calibration uses DEV.

- `anchor_memory`: restrict to anchors with the same source display name, falling back to all 64 if none exist. Retrieve the anchor whose combined option-text token set has maximum Jaccard similarity to the query's combined option-text set. Choose the query option with maximum Jaccard similarity to that anchor's delayed preferred option. Anchor ties use a fixed hash of base ID; option ties hash query base ID plus semantic option text.
- `anchor_profile`: the same user/global pool supplies a word-feature score: average token presence in preferred options minus average token presence in the other three options. Score each query option by the mean weight of its tokens. The same outcome-blind semantic-text hash resolves ties.

Both use Unicode case-folded alphanumeric token sets. Tie salt: `endo-pahf-cheap-control-tie-v1`. The rules are rotation-equivariant because ties depend on option text, not displayed answer letter. Output probabilities are the **actual one-hot policy**, never smoothed after observing errors. Infinite NLL from an incorrect deterministic action is represented as `null` with an explicit infinite-NLL count; the neural-superiority readout gate uses probabilities and never turns this into an artificial NLL advantage.

## Estimands and full decision

Retain full-vocabulary correct native answer-token probability U, target NLL, A/B/C/D-conditional probability/NLL, conditional argmax accuracy and answer mass. U assigns zero credit to every other vocabulary token. Average four rotations within task, then tasks equally. Report rotation strata and every leave-one-source-user-out aggregate; fewer than two source users is rejected rather than allowing a vacuous pass.

For every trained neural arm, the prescribed diagnostic requires minimum old-target-letter mean answer mass .05, paired mean semantic probability range ≤.20, maximum range ≤.50 and semantic-argmax disagreement fraction ≤.25. Pairing uses actual semantic indices under rotation. The global label-position averages remain insufficient for qualification. The no-update baseline must pass mass and paired probability-range checks, but `baseline_argmax_disagreement_required=false`: an uninformative uniform/tied stochastic distribution is a valid no-update comparator even though arbitrary deterministic tie-breaking disagrees across rotations. Its argmax disagreement is retained descriptively. This eligibility clarification was made prospectively, before any real endpoint; trained-arm/interface criteria are unchanged.

Train `raw_immediate` and `oracle_delayed` first, evaluate fixed final checkpoints, and stop after 70 updates unless all three comparisons qualify: oracle old target versus baseline; oracle old target versus raw; raw new target versus baseline. Each needs full-vocabulary NLL reduction ≥.10 and strictly positive gains in both full and conditional correct-choice probability. All three arms must pass diagnostics. Formatting-only acquisition cannot qualify.

Only then train the four sparse arms. A promising DEV method screen requires residual versus **each** raw, pooled SFT, pooled SDPO and mixture: mean delayed-target full NLL gain ≥.03, full correct-choice probability gain ≥.02, and conditional probability gain ≥.01. Full and conditional probability gains must be nonnegative in every rotation and leave-one-user-out stratum. Stratum NLL is reported but is **not an extra gate**; the approved rule gates aggregate NLL and stratified probabilities.

Residual must additionally beat **each** no-update, anchor-memory and anchor-profile readout by **more than .005 in both full and conditional correct-choice probability**. A cheap control within that practical margin stops the neural-superiority claim. This is not a statistical equivalence test. All arms/control results must be reported, including losses.

Strict probability inequalities use an explicitly serialized 1e-12 numerical tolerance: gains must exceed the floor plus that tolerance. This prevents floating-point roundoff at exactly zero or .005 from qualifying a format-only or tied result; it is not an effect-size relaxation. Non-strict scientific floors remain as written.

Interface/acquisition/diagnostic failure is invalid assay. A qualified assay missing any method/readout criterion is a scoped negative for this residual learner and budget; park the method claim. A pass is developmental support permitting proposed replication, never a paper green light. No CI/significance claim is inferred from DEV thresholds. Root's synthetic joint-rule operating-characteristic audit is a stress test, not neural power or a 5% false-positive guarantee; the thresholds are not tuned to its outputs.

## Compute, artifacts and limits

Full path: 210 optimizer updates; early acquisition stop: 70; preflight failure: zero. The full schedule entails 560 training forward calls, with 3,640 student and 3,640 teacher row presentations, plus 210 backward calls. Residual/mixture contain two student batches per update. Full evaluation is 128 interface rows plus seven 384-row DEV evaluations. Actual prompt/padded tokens, teacher/student forwards, backward tokens, optimizer updates and elapsed time are recorded separately. These counts are protocol arithmetic, not measured throughput. Equal labels/updates do not establish equal compute.

The 2–4 cached-GH200-hour estimate remains provisional. Setup, hashing, full raw-logit storage and I/O add cost; token lengths and throughput have not been measured for this new runner. Full-vocabulary float32 logits for 2,816 evaluation/interface rows may require several GB; plan storage/retrieval accordingly. No GPU is requested or launched by implementation/tests.

Execution authorization and the allocation ledger are recorded separately in `docs/GH200_COMPUTE_BUDGET_20260905.md`. The launcher defaults to a **14,400-second wall cap**, including input preparation, loading, hashing, all conditional arms and sealing. It also requires an explicit allocation-wide UTC deadline; the current allocation's ceiling is `2026-09-09T09:46:00Z`. The effective deadline is the earlier cap. This is an execution bound, not a changed scientific criterion.

On POSIX, the runner must own an isolated session/process group. Before model imports it starts one independent watchdog in a different session, waits for its atomically published PID/group acknowledgement, and saves `budget.json`. At 120 seconds before the hard deadline, the watchdog sends SIGTERM; cooperative checks preserve partial checkpoints and an invalid `REDUCED_BUDGET_STOP` receipt. At the hard deadline it sends SIGKILL to the runner's isolated process group before attempting receipt I/O, so blocked logging cannot postpone stopping compute. External termination requests also bound the remaining shutdown grace to at most 120 seconds. The watchdog survives SSH disconnection, exits after its actual parent exits, and refuses a shared process group. There is no additional per-run watchdog to configure. Allocation-wide cumulative accounting remains the transport's responsibility.

There is **no resume or automatic restart**. Interface failure still stops before training, acquisition failure still stops after 70 updates, and a budget interruption is incomplete rather than a method negative. Final checkpoint positions remain fixed. Every completed step reports elapsed time and current/cumulative-peak CUDA memory; these measurements will resolve the presently unmeasured throughput and activation-memory assumptions. BF16 model weights alone are approximately 18 GB; activation and optimizer memory require actual measurement, especially the residual/mixture objectives retaining two student graphs. A 97-GB GH200 is a plausible target, not an established memory guarantee. If the runner or GPU driver fails to respond even to OS process termination, host-level intervention remains necessary; a software timer cannot guarantee recovery from kernel failure.

Raw .npy logits, actual padded input IDs/masks, rendered-text hashes and exact row-to-batch indices permit independent arithmetic/tokenizer replay. A local immutable tokenizer can be replayed without model inference. Checkpoints, step logs, effective model/optimizer/environment/source settings, complete input bindings, label-access receipts and full decisions are recursively sealed. Source closure and actual Git state are recorded; hashes do not independently prove that neural forwards used those weights. The verifier distinguishes byte/arithmetic/tokenizer replay from neural recomputation.

Implementation: `src/interaction_sprint/hindsight_pahf_reduced.py`; launcher: `scripts/run_hindsight_pahf_reduced_dev.py`; verifier/integrity helpers are separately reviewed. CPU tests cover schedule coverage, nonanchor-label exclusion, exact IDs/rotations/users, complete interface grid, format-only acquisition/method failures, user/rotation cancellation, cheap-control matching, transition identity and one-hot rotation-equivariant readouts. The launcher has not been smoke-run with a model. Remote package/library availability, CUDA memory, execution throughput and numerical stability remain unverified until separately authorized qualification.
