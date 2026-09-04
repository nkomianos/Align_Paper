# Full assays and evidence-dependent follow-up

The user's original #1 is Hindsight Is Not Counterfactual; #2 is Dialogue
Retractions as Algebra. User authorized more experiments on September 3 PDT /
September 4 UTC. We ran the existing frozen full assays unchanged, then designed
the following separate developmental audit. No threshold was changed afterward.

## Completed full assays

512 hindsight forwards: anchor-only initial-choice probability 0.994064;
truthful-correction probability 0.938036. The wrong-action hindsight advantage
is +2.765316 nats, but the simple anchor prompt restores only **0.003142
percentage points** on average, below its registered 10-point criterion. It
does not establish an effective correction. All overlapping smoke probability
records reproduced exactly; the average changed with the additional cases.
Decision: `NO_SIGNAL_IN_THIS_PROMPT_ASSAY_DO_NOT_INFER_THESIS_FALSE`. Park this
simple prompt-correction route. This does not disprove non-identifiability or
test an actually trained causal estimator/SDPO policy.

768 UNDO forwards:

| Updates | Edited-history accuracy | Matched-depth control | Canonical | Counterfactual |
| --- | ---: | ---: | ---: | ---: |
| 4 | 100% | 100% | 100% | 100% |
| 20 | 97.92% | 100% | 100% | 100% |
| 60 | 68.75% | 100% | 100% | 97.92% |
| 100 | 66.67% | 100% | 100% | 100% |

Stale choices occur in 43.75%/50% of cancellation-or-overwrite cases at depths
60/100. This is a specific signal in this apparatus, not a claim across all
conversation types. Decision: `RESIDUE_SIGNAL_DESIGN_TRAINING_NEXT`.

Both archives were retrieved and locally verified with committed source.
Evidence: `retrieved/interaction_full_20260904T0348Z`; remote/local archive
SHA-256 `cbb8a5e9ecc2e0c19b87bd589fb41919841ea56fdc8452e8236fbd0d93c63d7a`.
Raw artifacts and earlier attempts are preserved. No weights were updated.

## Queued immediate follow-up: UNDO-Audit

Implementation `src/interaction_sprint/undo_audit.py`; seed 9040349. Four
surfaces, two update types, three lengths, six fresh histories per cell, six
arms: **864 forwards**. Two command renderers, randomized field names/noise
values, and middle versus late consequential updates. No reuse of original
histories, but this is **post-pilot developmental testing**, not a pristine
confirmatory dataset.

Arms: canonical state; ordinary history; matched-operation-count padding;
explicit final update (including literal UNSET); answer-neutral final reminder;
and a one-update counterfactual. The explicit update condition changes the
language of the consequential operation, not its intended semantics. The
reminder contains no instance-specific answer. Canonicalization is an executable
state-tracking baseline, not a novel proposed model method. Padding is matched
on update count, not exact tokens. No held-out human conversation is involved.

Before inference, freeze these descriptive triage rules:

* Probability mass >=0.5 and canonical/padding/counterfactual accuracy >=0.9 in
  each relation/depth/renderer cell, otherwise the assay is invalid.
* Long-history aggregate gap >=10 percentage points versus padding is required
  for a fresh effect. Subgroup results are retained, not silently averaged away.
* If either reminder or explicit-update normalization comes within five points
  of padding, report `SIMPLE_BASELINE_SUFFICIENT_DO_NOT_CLAIM_TRAINING_NEEDED`.
  This does not erase the failure, but undercuts a need for the proposed training.
* Only a robust effect not cheaply resolved progresses to independent-family
  replication and a baseline-matched training design.

These cutoffs are engineering/PI screening rules, not significance tests. A
positive outcome does not guarantee an ICLR-worthy contribution.

## Subsequent queue, not yet launch-ready

See [machine-readable follow-up queue](../configs/interaction_followup_queue_20260904.json).
Independent-family validation must first select a public, pinned model and pass
format/capability controls. A different Qwen size alone is not cross-family
replication. No such model is selected or downloaded yet.

If warranted, the training study will compare local-relation learning against
generic short-history SFT and canonical-context distillation with matched data,
updates and compute, plus inference-only reminders/normalization/state tracking.
Train at 2–5 edits; evaluate fresh 20/60/100-edit histories across three seeds,
with unseen templates and domains. The training method is not yet implemented.
Method success requires meaningful improvement over strong baselines without
losing short-context or counterfactual competence—not merely beating the
unmodified model. A useful, non-vacuous theory or compelling generalization
result and a renewed novelty check are required before a paper green light.
