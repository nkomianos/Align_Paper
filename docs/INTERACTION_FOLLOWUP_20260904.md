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

## Additional collision check while the audit runs

[ICF-Bench, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b13d00a62d438856cfe6fbd13b6b2cb8-Abstract-Conference.html)
already studies in-context forgetting, subtask revision and changing preferences.
Its [released paper](https://openreview.net/pdf?id=hcJywRYc3n) and
[repository](https://github.com/qianyuli123/ICF-Bench) are mandatory related work
and potential external validation, not new datasets downloaded in this turn.
The observation that old instructions survive removal is therefore not itself
a new contribution. The novelty burden is a useful compositional training result
or a stronger mechanism/theorem, with transfer beyond our generated registers.

[CCOPD](https://arxiv.org/abs/2605.30251) already distills canonical-context
behavior into multi-turn trajectories, and
[curriculum RL for multi-turn conversations](https://aclanthology.org/2026.acl-long.1540/)
is another relevant learning baseline. This further rules out framing generic
history training as new. No abstract or paper submission is justified by the
current single-model generated-data result alone.

## Completed UNDO-Audit result

All 864 forwards completed and were locally verified. Long-history results
(96 matched histories per arm across depths 60/100): ordinary history **87/96
= 90.625%**, canonical/padding/counterfactual **96/96**, reminder **87/96**,
explicit-update normalization **91/96 = 94.792%**. The ordinary-history gap is
**9.375 percentage points**, below the frozen 10-point rule. All validity
prerequisites pass. Verifier decision: `NO_FRESH_HISTORY_SIGNAL_PARK`.

This is not zero effect, nor proof of a universal absence. The gate misses by
one case at this sample size; we must not call 9.375% qualitatively different
from 10% or silently lower the threshold. The earlier 31–33-point magnitude did
not persist in the aggregate of this new setup. Wording, noise content and update
position changed together, so this audit does not identify which factor caused
the difference. Reminder did not solve the errors; explicit normalization helped
but is not perfect. No learning-method success has been demonstrated.

Next queue entry is a **paired interface-factorial diagnostic to design**, not
an already coded or running experiment. It would isolate those factors without
reclassifying either completed decision. Cross-family replication and training
remain held behind robustness and a meaningful novelty argument. In particular,
nearly perfect short-history accuracy also raises a training-design question:
ordinary short-example answer SFT may provide very little corrective gradient.
We should not assume it will automatically fix long histories.

Evidence: `retrieved/undo_audit_20260904T0354Z/undo_audit_20260904T0350Z`;
source `18e6207b1fee7c694128409f10616c06968f88f6`. Remote/local archive SHA-256
`817d2b95c6c3b9cfccd76156c3f9546ededa81e7e1e95571acfdbe153c173f93` agrees.
All earlier evidence is preserved. 31 relevant CPU tests pass. This turn ran
**2,144 additional model forwards** (512 + 768 + 864), no training. GPU idle
after the completed audit; no automatic training/replication started.
