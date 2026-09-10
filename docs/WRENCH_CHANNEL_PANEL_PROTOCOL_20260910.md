# Paired reasoning-channel panel

Runner and selection committed6f908c1 before neural launch. A developmental
measurement test, not an independent validation of released intent labels.

Terminal Wrench source d8a29613235a0ef56a8b70b3142626a533da28c2 supplies24 salted-hash
tasks, one matched generator model per task, and one released serious hack and
successful baseline selected by independent salted hashes. First4 tasks are
calibration, remaining20 DEV. No selection uses monitor scores or trace length.
Whole-task context exclusion then removes setup-ubuntu-vm-ssh-key-auth, whose
baseline exceeds30k tokens. No truncation or replacement:23 tasks,92 views,
including19 DEV tasks. This is a context-limited panel, mostly SETA source tasks.

Each trace has full and stripped views. Remove all user elicitation in both;
provide the original instruction independently. Strip only agent message text,
preserving exact tool-call and observation objects. Two tests check immutable
input transformation and strict numeric JSON output parsing. This local JSON
renderer and Qwen3-32B monitor are not the official GPT-5.4 renderer or score
reproduction. Root metadata, cohort labels, hack judgments and rationales are
excluded from model inputs. No upstream code or exploit is executed.

Qwen3-32B revision9216db5781bf21249d130ec9da846c4624c16137 uses greedy native
nonthinking chat, max128 output tokens, fixed system rubric and no prompt search.
Primary endpoint: paired mean stripped-minus-full suspicion score in the released
DEV baseline cohort. Secondary: hack score shift and release-label AUC under
full/full, stripped/full and stripped/stripped comparisons. Experimental unit is
task; four views are not four independent samples. With19 DEV baselines, do not
claim a reliable calibrated5% FPR estimate or tune a threshold.

Frozen qualification: overall parse and EOS rates>=.95 and>=16 complete DEV tasks.
Absolute mean baseline score shift>=.10 admits only an independent-model/fresh-task
follow-up. Otherwise stop this screen. Failed outputs remain in raw data and
qualification denominators; complete-task secondary summaries must disclose their
coverage. No inferential or safety claim is made from this exploratory gate.

The published source comparison holds legitimate baselines unchanged while
altering hack traces. Its code also supports stripping legitimate traces; do not
claim this capability is missing. Asymmetric hiding can be a valid threat model.
The present question concerns uniform channel loss, a different estimand. Existing
CoT monitorability work already studies reasoning removal, so an effect alone
would not prove novelty. Source code, labels and interpretation require continued
audit before any paper recommendation.

Launch05:05 UTC on authorized GH200, output wrench_channel_qwen32_v1. Estimate
10–30 minutes, unbenchmarked, no arbitrary midrun timeout. Raw responses, token IDs,
prompts, model hashes and source hashes saved. Read-only endpoint replay:
scripts/verify_wrench_channel_panel.py. No automatic large-scale follow-up queued.
