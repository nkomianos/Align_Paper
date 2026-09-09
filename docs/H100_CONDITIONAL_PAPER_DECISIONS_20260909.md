# Conditional research decisions during the H100 run

Prepared while Hindsight calibration was live, before compensation/reference
outcomes. Existing pilot gates and data stay unchanged. This memo restricts
interpretation and follow-up; it does not convert exploratory data to confirmation.

## Literature recheck

[Glass et al.](https://arxiv.org/html/2608.24988), sections 4–5, already distinguish
behavioral recovery from reversal of embedded weight edits across multiple
models. Their weight-projection evidence does not itself locate a learned
compensator. Repeating behavioral recovery plus persistent edit is insufficient.
Our potentially distinct question is whether adaptation creates a causal
dependence on the edit and whether targeted removal of the adaptation repairs
withdrawal damage while preserving utility. This remains a hypothesis.

[Reference-Grafting](https://arxiv.org/html/2608.29458), discussion and limitations,
already requires a trustworthy reference with the same suppression mechanism,
reports failure across mechanisms, and leaves unrestricted generation unresolved.
Our prompted mean/median screen supplies neither reference acquisition nor a
new elicitation method. Even a positive result only qualifies an apparatus for
studying reference reliability. It must not become a generic sandbagging paper.

## Independent code findings

Inspected data.py, compensation.py, reference.py, neural.py and their frozen
implementation contract. The frozen pilots use simple synthetic permission
equality, not rich safety behavior. Each split has one wording template and
32 base scenarios, with two option orders each. Resource identifiers are unique;
this establishes neither 32 task families nor real-world generalization.

Compensation's route uses absolute accuracy differences after withdrawal. This
does not prove a directional rebound opposite the original edit; aggregate
accuracy also hides which semantic class changes. The retained route is a
screen only. Inspect signed per-base choice-probability changes and option-order
consistency before proposing the claimed mechanism. Any post-hoc calculation is
descriptive and must be prospectively confirmed on new data.

The random edit is norm-matched in parameter space but not effect-matched.
Its recovery and withdrawal contrasts cannot eliminate generic damage as an
explanation. The implementation records an adapted dose sweep but no complete
matched unadapted/no-edit-trained sweep. One trainable site cannot demonstrate
distributed circuit reorganization. No method benefit over simply removing E
or revalidating/retraining has yet been tested.

Reference chooses between mean and median on DEV at budget 16/corruption 25%.
That comparison is model selection, not two independent confirmations. Its
reference uses gold paired honest captures and the withholding instruction is
explicit. It cannot establish deceptive intent or discovery of a hidden goal.

## Conditional queue after saved-output and supplemental verification

| Stage outcome | Next action |
|---|---|
| Hindsight teacher fails | Invalid interface; no causal-method expansion or prompt tuning on the same endpoint. |
| Hindsight SFT acquisition fails | Audit train/held-out loss and option-order behavior; stop sparse-label method expansion. |
| SFT and frozen pass, updating teacher fails | Scope to first-token target dynamics. Require a distinct external task and comparisons with published self-distillation controls before training more arms. |
| All Hindsight arms pass | Apparatus only. Sparse trusted-label study remains conditional on a novel, decision-relevant contribution and fair mixture/SFT baselines. |
| Compensation formation/recovery fails | Identify the failed prerequisite. At most one isolated developmental repair; do not count failure as absence of compensation. |
| Compensation qualified withdrawal fails | Stop this candidate/version; no seed expansion to search for a positive. |
| Compensation screen passes | First analyze signed per-base changes, label-order effects and utility; then freeze a matched-effect factorial and causal repair experiment on new cases. |
| Reference screen passes | Require reference acquisition/qualification without privileged unlock captures and a trained organism before expansion. |
| All candidates fail or remain scientifically indistinct | Preserve negative evidence and revise the paper recommendation. Do not manufacture a manuscript or spend the remaining rental on duplicate screens. |

No secondary-family or replication job is admitted by this memo alone. A useful
follow-up must state data, intervention, baseline, primary endpoint, scientific
kill rule and measured runtime. Estimate against the current 12-hour H100
allocation; the possible future GH200 allocation is not counted as available.

## Actual paper path

1. Finish and independently verify the prepared pilots; choose a contribution
   only if it survives the strong baseline and novelty objections above.
2. For the selected candidate freeze new development/confirmation data and an
   equal-information comparison, including a simple alternative that could win.
3. Establish the controlled effect and a useful remedy, then independently
   replicate across training seeds and a second model family. Treat base tasks
   as clusters and report seed variation separately.
4. Add external validation with an accurately stated estimand. Synthetic
   permission equality or an explicit withholding instruction is insufficient.
5. Build manuscript figures from immutable verified artifacts, specify limitations
   and failure cases, reproduce the tables, and only then assess submission readiness.

The strongest conditional new thesis is learned dependence on a persistent
intervention with a targeted repair. There is currently no empirical green light
for that thesis and no high-confidence acceptance claim.
