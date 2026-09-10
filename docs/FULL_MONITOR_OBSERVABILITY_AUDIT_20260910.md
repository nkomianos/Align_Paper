# Full-release monitoring observability audit

This extends the earlier selected214-row overlap cohort to all4472 released
CheatBench rows, pinned at461360b34e93644955563dcc9243c6cc1a53bd0f.
Raw parquet SHA256b4d2a1496e5f8de7b4c0e4791bd9f8fcbd0a3567f95d2e759b5da20ebb160e63.
Reproduction: scripts/audit_full_monitor_observability.py. Full records and hashes
are in artifacts/monitor_external_sources_20260909/FULL_OBSERVABILITY_AUDIT_V1.json.

## What is established

Our existing strict observer projection accepts3196 rows. It rejects544 Cybench
retrospective reports,218 HAL-USACO rows with unsupported event schemas,194 rows
without conversation events, and320 rows with unresolved user-message references.
These are projection/admission categories, not544+218+194+320 mislabeled examples.
Unsupported schema means our adapter lacks support; retrospective reports may
be intentional for the benchmark's retrospective task. Neither category can be
described as intrinsically unusable or mislabeled without a matching estimand.

All194 eventless rows are labeled benign. Of320 unresolved-user-reference rows,
299 are benign,20 attempt and1 cheating. The release README explicitly includes
unauditable traces within benign. Consequently this label does not, by itself,
certify a fully observed safe action trajectory for our proposed monitor task.
The issue is a mismatch with a safety estimand, not a hidden contradiction of
the release's disclosed label definition.

Two pairs have byte-identical complete `trace` field strings and different labels:

| IDs | Labels | Trace SHA256 |
|---|---|---|
| cb-002029 / cb-004470 | benign / cheating | 4f7fca0343ada636cf964af1f532ff5b545be750dcc005394868bab4946fb21f |
| cb-002287 / cb-004472 | benign / cheating | cc057a1abb6de3f5a0d6f2cd259d2004f32898369beecedb14eea9a597337115 |

The benchmark and harness match; model metadata differs as gpt-5.4 versus
gpt-5.4@openai. Positive rows add harness-leak rationales and localization, whereas
negative rows say no cheating detected. No independent judgment of which label
is correct is made here. The identity is exact for the trace field, not all row
metadata. A deterministic trace-only classifier necessarily disagrees with at
least one label in each pair under the release labels. That is only two rows
out of4472, not evidence that most benchmark errors or model rankings are explained.
External privileged context or annotation revisions may explain the disagreement;
we have not reconstructed the annotation history. Do not silently deduplicate or
relabel. This finding differs from the earlier nonidentical repeated-trial pair.

## Upstream source recovery

Full cohort contains583 rows attributed to yoonholee/terminalbench-trajectories,
covering529 trial-name/model keys.394 match an original event stream exactly under
the previously declared null-field/index normalization.189 matching-name original
rows deserialize `steps` to JSON null. The old identity script crashed on those;
it now records them separately rather than treating null as a real empty stream.
The189 unmatched cases therefore do not demonstrate fabricated or modified events.

There are54 repeated trial-name/model keys.320 released unresolved references
match that defect in the original source;54 have a unique candidate with fuller
events and a UUID. This does not prove episode identity without retained-field
and metadata checking. Report: terminalbench_original/FULL_UPSTREAM_IDENTITY_V1.json.

Wider replay exposed a bug in our recovery checker: a dictionary keyed only by
steps SHA256 could overwrite another trial having the same text. The checker now
retains all matches and filters by trial, model and agent, plus candidate UUID.
Ambiguous identities fail admission. Two dedicated identity tests and three
retained-field tests pass. The first failed full-recovery directory is preserved.

Corrected full recovery:54 candidates,18 retained-field-compatible,36 with
unexplained differences; none failed the final identity check. Compatible files
are saved separately under terminalbench_full_recovery_v2. Replaying the original
20-candidate cohort still gives8 compatible and12 incompatible, so that earlier
summary survives. These are potential context recoveries, not validated labels,
independent tasks, or neural findings. Full recovered traces still require review.

## Paper decision

This audit improves evidence integrity and identifies concrete measurement
problems. It does not yet establish an ICLR contribution. A paper would need a
distinct question beyond known authorization-relative monitoring, defensible
labels, and evidence that the problem materially affects conclusions across
independent sources. Generic missing-context impossibility is not novel; two
conflicting-label pairs alone are insufficient. No neural run is admitted solely
because these structural anomalies exist.

## Diversity and nuisance-feature follow-up

All18 compatible fuller candidates are large-scale-text-editing episodes across
four models and two harnesses. They add no new underlying task. Receipt:
FULL_RECOVERY_DIVERSITY.json. This rules out treating recovered episode count as
eighteen-task replication, while preserving their usefulness for context review.

audit_monitor_metadata_baseline.py fits logistic regression using benchmark,
harness, model and schema only, with five stratified folds grouped by exact raw
trace hash. The label here merges cheating and attempt versus benign; it is not
the release's cheating-only binary task. At19.79% positive prevalence, out-of-fold
AUROC=.93508, average precision=.86259 and balanced accuracy at.5=.86305.
Report METADATA_BASELINE_V1.json saves every fold and prediction. No trace text,
rationale, localization or gold annotation fields enter the features. Exact
duplicates cannot cross folds, but tasks, models and sources can. This demonstrates
source-label association on a mixed-source split, not cross-source generalization,
an official benchmark comparison, neural shortcut reliance or a new safety result.
No hyperparameter search or threshold tuning was performed.

Full-corpus projected-view preparation is running locally to test instruction
and context-length admission across intact data. Do not infer that prepared
schemas have certified labels. Subsequent matching and semantic review remain.

## Held-out benchmark transfer

audit_monitor_metadata_transfer.py fixes the same four categorical features and
logistic regression C=1, with no tuning. Each of nine benchmark names is held out
in turn; exact raw-trace hashes found in its test set are also removed from
training. Positive still merges cheating and attempt. Full predictions and fold
membership are saved in METADATA_TRANSFER_V1.json.

Mean within-benchmark AUROC is.52046 over the eight two-class cohorts; HAL-USACO
contains only positives, so its AUROC is undefined. Pooled out-of-fold AUROC is
.46869, but pools predictions from different fitted models and must be interpreted
accordingly. At the fixed.5 threshold, the model predicts all Terminal-Bench rows
positive and all other held-out benchmark rows negative. This is not a useful
transferring detector. The earlier mixed-source.93508 AUROC should be described
as source-specific association, not robust monitoring performance.

Holding out a benchmark name is not proof of disjoint underlying source corpora.
No neural monitor was tested by this calculation, and no released label was
independently certified. This closes the metadata-only baseline's generalization
claim; it does not establish that every text-based model has the same failure.
