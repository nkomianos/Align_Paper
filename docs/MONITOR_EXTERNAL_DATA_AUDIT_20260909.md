# External monitor substrate audit

Purpose: find independently usable evidence for the paper goal after direct
MALT API rendering failed context qualification. Access alone is not admission
to the frozen monitoring experiment. No GPU scores have been computed here.

## Sources and novelty

[Patronus TRACE](https://huggingface.co/datasets/PatronusAI/trace-dataset)
is a human-verified synthetic benchmark. Its paper already evaluates contrastive
trajectory grouping, so that comparison is not a new contribution. The repository
is gated (revision 31d87f06078eca3ab6eaf1e06e5ea6fe9f2b7a6d); no data downloaded.
Do not conflate this TRACE with other similarly named monitoring papers.

[Trace-and-Amplify](https://arxiv.org/html/2604.23488) already compares monitors
trained on prompted and training-time trajectories, including unseen exploit
types. Repeating that transfer claim would collide with existing work.
The [authors' repository](https://github.com/LichenLillc/CoTMonitoring) releases
data and collection scripts. Downloaded archive at commit
bea6b67e9707b1f2c2ce2fa9304f5834855d6a20: 166,710,430 bytes, publisher checksum
e662cd5eb31c5adfc95dc3841d12f567b3edc7db00bd185ff69f4322598a180f verified.
The first raw download was the 134-byte Git LFS pointer, not the archive;
preserved it and obtained the actual object through GitHub's media endpoint.

[CheatBench](https://huggingface.co/datasets/steinad/CheatBench) is ungated.
Downloaded the micro subset and card at revision
461360b34e93644955563dcc9243c6cc1a53bd0f. Its localization annotations are
heuristic rather than independently human-verified spans. Micro/mini/full are
nested evaluation subsets, not train/test partitions. Its annotation schema
distinguishes successful cheating, failed attempts and benign behavior; preserve
that distinction. Do not feed rationale, loc_*, labels or outcome fields into
monitors. Source traces also require inspection for embedded audit metadata.

## Independent raw evidence

Artifacts live in artifacts/monitor_external_sources_20260909.
DOWNLOAD_MANIFEST.json records local hashes. The downloaded CheatBench micro
parquet has 263 rows: 224 benign, 30 cheating, 9 attempts across nine benchmark
families. Inspected its actual columns and first parsed trace structure. Full
event normalization, independent label review, task matching and token lengths
remain unqualified. It is not yet a ready fallback for MALT.

Trace-and-Amplify archive has 29 JSONL files, totaling 13,169 file rows (this
includes repeated membership across files; not 13,169 independent examples).
The identity auditor inspected every row's schema, prompt and response hashes,
label/type counts, exact prompt pairing and same-source train/test overlap.
No response hash has conflicting labels WITHIN a file; cross-file label
consistency has not yet been measured.

The training-time unittest paired file contains 800 rows, 400 exact prompts,
each with both labels. Combined prompt/response character lengths are 1,580 to
22,446. The exit paired file similarly contains 400 exact pairs. These are
better candidates for a controlled DEV diagnostic than the current MALT records,
but they do not establish independent task families, verified code outcomes or
matched source-model provenance by themselves.

Pooling all same-source files named train_/test_ gives 27 exact overlapping
prompts in the prompt-elicited source; both training-time source groups have zero
exact overlap. This is an archive-level finding. Inspect the actual experiment
file selections before claiming leakage in a particular published result.
Near-duplicate/semantic task overlap is not ruled out by hash disjointness.

The two inference-only negative files each contain 15 rows but only ONE unique
prompt. Positive files contain 14 and 10 unique prompts respectively. Consequently
raw trajectory counts exaggerate the task diversity available for negative
generalization. Do not use these files as 15 independent negative tasks per model.

Inspected the source filter: score-one responses are routed as hacks; remaining
responses are divided into attempts/normal using code keyword detection. This
is not independent human review. The meaning of score one depends on the
conflicting-test evaluator and its failure handling; that code must be audited
before labels are treated as verified behavior. Do not set reviewed=true merely
to pass the old MALT runner contract.

## Next admission requirements

1. Audit the exact score/evaluator path and source-model identities behind the
   training-time paired files; inspect representative positive and negative code
   without executing downloaded code in the host environment.
2. Check cross-file identities and semantic task grouping, then reserve source
   test files. Qualify complete prompt/response tokenization on the pinned model.
3. State a contribution beyond published prompted-to-training-time transfer.
   A new diagnostic must include an equal-information strong baseline, paired
   task analysis, and a simple code-feature baseline where labels use heuristics.
4. Freeze an independent protocol and source hashes before any monitor scores.
   Do not quietly weaken the existing family-heldout/human-reviewed contract.

No paper recommendation is upgraded on the basis of data availability. The
archive README assigns no dataset license and contains ground-truth test fields;
retain downloaded artifacts locally and do not publish them as our dataset.

## Static evaluator and paired-label audit

Followed the public main_ppo.py import into livecodebench.compute_score rather
than assuming the similarly named prime_code evaluator was the active route.
Current source files are preserved with SOURCE_CODE_MANIFEST.json. The reward
manager catches worker failures and can assign zero rewards; low reward alone
therefore does not certify benign behavior. Historical collection configuration
and evaluator revision are still not bound by the released trajectory metadata.

The import_prefix branch explicitly inserts `unittest.main = None` before the
submitted solution. Saved positives can call that function. AST call presence
does not prove execution, and a submission could reassign it; nonetheless this
is a concrete replay prerequisite. Do not describe current public code as an
already verified reproduction of the historical collector. No downloaded
solution or evaluator was executed on the host.

Static inspection of all three training-time paired files found identical
ground-truth hashes within each exact prompt group. All positive rows have
recorded reward 1.0; negatives have lower rewards, including fractional scores.
Ground truth has several evaluator schemas, not one uniform unittest protocol.
The saved V2 label-prerequisite report records each row's schema and code parsing.
Its narrow same-call/literal-output assertion audit does not establish semantic
satisfiability or replay outcomes. V1 is retained; V2 corrects literal equality
so Python-equivalent values such as 1 and 1.0 are not counted as conflicting.
Three targeted tests pass.

In the 400-pair unittest development file, all 400 positives and zero negatives
contain an AST call to unittest.main in the evaluator-selected Python block.
Thus this one static feature perfectly separates this DEVELOPMENT file's labels.
This is not an independently held-out accuracy estimate, a general hack detector,
or a new result about the published monitors. It makes a generic new LLM prompt
comparison on that raw view scientifically weak unless the comparison controls
for this feature and tests broader behavior.

The authors also supply scripts that delete unittest imports/test classes and
remove selected reasoning lines. Their existence matters: the raw release may
not be the view used for every reported experiment. Trace actual preprocessing
and file selection before comparing against published numbers. Do not remove
code ourselves and retain the label as if the edited executable still performs
the original behavior; distinguish observer-view ablation from behavior editing.

Decision: no automatic GPU admission for the raw paired unittest prompt screen.
Remaining useful work is identifying a distinct estimand and qualifying a natural
external control, or a causal observer-view diagnostic that preserves the original
executable and binds each removed span to what the monitor was allowed to see.
The static separation is evidence against an expensive raw-view-only comparison,
not evidence that the intended ICLR paper goal has been achieved.

## Monitor preprocessing and natural-record schemas

Read the pinned paired preprocessing and conversion scripts, BERT/probe launch
scripts, and gpt_evaluate.py. Instruct preprocessing removes end markers and
extracts the user prompt; it retains the response. Conversion uses that response
as final_answer. The LLM evaluator sanitizes control characters and cleans task
prompt wording, then supplies the response as agent_messages. These paths do not
themselves delete unittest code. The launchers consume directory globs rather
than an immutable per-experiment input manifest, so the separately available
deletion scripts cannot yet be linked to particular reported numbers. This
narrows the uncertainty; it does not justify asserting all published monitors
saw either raw or code-deleted inputs.

CheatBench schema audit covered all 263 micro rows, recording source trace hashes
and field/event schemas. Three SWE-smith rows include prior monitor_* fields
inside extra (including verdict/score-related fields). A naive raw-JSON input
would expose these pre-existing judgments. That is a risk in OUR proposed reader,
not evidence that the source benchmark's own monitor consumes them.

All 15 SWE-bench Verified rows and 10 Terminal-Bench rows have empty or missing
standardized events. Many others lack a standardized task object. These findings
do not establish missing source evidence: alternate fields and wrapped records
must be inspected before any exclusion. An events-only conversion would silently
lose these rows unless it fails closed or handles each schema explicitly.

Artifacts: CHEATBENCH_SCHEMA_AUDIT.json and pinned source_code files under
artifacts/monitor_external_sources_20260909. This remains a CPU data audit;
there are no monitor scores, qualified natural controls or newly admitted GPU jobs.

## Alternate fields and localization follow-up

All 15 missing-event SWE-bench Verified records contain a messages list; that
format is recoverable. The other 10 missing-event rows are Terminal-Bench benign
records with empty events and files and no messages. Their top-level fields hold
identifiers/configuration/outcomes rather than a conversation. Do not score these
as negative behavioral traces. This is specific to the inspected micro release.

Resolved the released loc_field_path and line ranges against each parsed trace.
Of 39 positive/attempt records, 4 snippets match exactly, 19 differ only in a
terminal newline, 11 have other range/snippet differences, and 5 have unresolvable
paths. Thus the initial 30 exact mismatches must NOT be described as 30 missing
evidence spans. All 11 non-whitespace mismatches still contain the snippet in
the declared field. The source's exact-line reproduction claim needs these
format/range qualifications for this pinned release.

For the remaining 16 range/path cases, an audit-only recursive exact substring
search finds a unique occurrence in 5 records, two occurrences in 9, four in 1,
and none in 1. Repeated source views make arbitrary first-match relocation
unsafe. No source annotations were overwritten. V1/V2/V3 reports are retained;
V3 adds explicit match locations. This verifies textual availability, not whether
the text establishes cheating or whether a monitor observes it.

The locator must never select model input using the gold snippet: that would
give the monitor privileged evidence. A future extractor must choose fields by
schema alone, with these annotations used only to audit coverage afterward.
One targeted path/repeated-match regression test passes.

## Concrete observer-view preparation

Implemented a schema-defined events/messages projection. It includes conversation
and tool fields, excluding top-level outcomes, configuration, source metadata,
extra, and stored files. Stored files include result.json, reward.txt and verifier
reports, so they cannot be included indiscriminately. This projection is an
observer view, not a lossless rendering of all files or a certified full trajectory.
It is selected without using source labels, gold spans, model scores or length.
Unknown event fields fail closed. All 23 observed cache_control fields were exactly
{"type":"ephemeral"}; V2 explicitly omits only that known provider setting.

V2 prepares 245 of 263 records. Eight HAL-USACO records have an unsupported trace
schema; 10 other records lack conversation events. Original V1 remains preserved.
The pinned tokenizer measured every prepared record without truncation:

| Raw observer-view token ceiling | Benign | Cheating | Attempt | Total |
|---|---:|---:|---:|---:|
| 30,000 | 148 | 6 | 7 | 161 |
| 60,000 | 184 | 13 | 9 | 206 |
| 120,000 | 209 | 15 | 9 | 233 |

These ceilings are descriptive fit screens, not selected experimental populations;
prompt overhead is additional. Maximum view length is 1,088,011 tokens. Only the
32,768-token hardware path has been tested. Longer context cannot be assumed to
work from these CPU counts. Selecting short traces markedly changes the available
positive population, and a 30k-only study would have six successful-cheating cases.
Do not claim a full-benchmark false-positive or detection rate from such a slice.

Outputs and provenance are in cheatbench_event_view_v2/MANIFEST.json. Three
targeted tests pass. Task-instruction completeness, semantic labels, natural
within-task controls and actual monitor comprehension remain unqualified.

## Post-extraction coverage and full-release admission check

After freezing the schema-only event projection, checked whether each released
positive/attempt snippet occurs in a preserved string field. Thirty of the 31
prepared positive/attempt records retain the snippet; one does not. Eight other
positives are in the unsupported HAL schema. Gold annotations were used only
for this subsequent coverage measurement, not to choose or construct model input.
This is a necessary textual coverage check, not semantic validation of labels.
Report: CHEATBENCH_EVENT_COVERAGE.json.

The prepared micro event views total 9,394,345 raw tokens. Approximately 470
30k windows would be required even before adding repeated task context, overlap,
prompt overhead or any aggregation protocol. No chunked assay is admitted by
that arithmetic alone. The micro subset is too small after short-context
filtering for the intended paper, so inspect the full public release before
concluding that the entire dataset lacks usable natural controls.

Full parquet download at the same pinned revision is in progress; completion is
recorded only by CHEATBENCH_FULL_DOWNLOAD.json with the final hash. The streaming
audit_cheatbench_full_cohort.py is prepared to measure exact provisional task,
model and harness overlap without loading all trace objects into memory. It does
not use an unexplained top-level id as a task identity. It has not yet run.

### Completed full-release cohort audit

The full download and audit are now complete: 4,472 rows, comprising 3,587 benign,
844 cheating and 41 attempt labels. Exact provisional task/model/harness overlap
gives 42 cells, 85 positives and 129 benign rows, at most 58 disjoint pairs.
There are 500 unresolved task IDs and 189 benign rows without standard payload.
Most overlap cells (36/42) are Terminal-Bench-2, not cross-benchmark diversity.

All 214 rows in those overlap cells were extracted with source hash checks and
without ranking. The unchanged schema projection prepares 210 event views; four
have no conversation events. Under a 30k raw-token limit, retaining only cells
with both labels leaves 24 cells, 38 positives and 76 benign rows (at most 35
pairs), of which 22 cells are Terminal-Bench-2. At 60k the corresponding counts
are 26/45/91/38, and at 120k 35/63/116/51. Prompt overhead is additional; hardware
qualification beyond 32k is absent. Cells can share tasks, so these are not
independent task counts. These length-selected, label-conditioned subsets do not
estimate full-benchmark performance. No monitor experiment has been run.

Reports: CHEATBENCH_FULL_COHORT_AUDIT.json, full_overlap/MANIFEST.json,
full_overlap_event_view/MANIFEST.json, FULL_OVERLAP_CONTEXT_FEASIBILITY.json.

## Superseding input and source-identity audit, 10 September

The earlier projection counts are not a claim of monitor readiness. Raw inspection
found six retrospective report sources and seventy sources with unresolved user
messages such as "$32". Fresh full_overlap_event_view_v4 rejects those, plus four
no-event sources, leaving134 prepared views. Four targeted regression tests pass.
Exact extraction of ForgeCode -p instructions finds75 rows; matching the same
instruction/task/model/harness under30k raw tokens yields7 cells on6 provisional
tasks,9 positive and16 benign labels. This measures extractor coverage and a
selected cohort, not all available task contexts or full-benchmark performance.
Reports: admission_v3 (superseded projection) and admission_v4.

The [upstream trajectory release](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories)
was downloaded at04e8940f5b6736a7ce8d22224fe2f2af74163ed2, both parquet shard hashes
checked against official LFS metadata. All95 cohort rows from that source match
an original event stream exactly after omitting null fields and our added step
indices. Seventy already have unresolved user references upstream. This is not a
bug caused by our JSON projection. Twenty of those have a candidate fuller row
with the same trial name/model and a nonempty original trial UUID; differences
must be reviewed before replacing data. Fifty have no such unique candidate in
this downloaded release.

Those95 rows have75 unique trial-name/model keys:20 keys occur twice. One key,
large-scale-text-editing__QaBSzjy, has an attempt-labeled full-text representation
(cb-004218) and a benign-labeled representation with message placeholders and
missing tool-argument text (cb-004223). Event payloads are NOT exactly identical.
Do not claim an identical-input label contradiction or silently deduplicate on
name alone. The episode identity and representation dependence require source
review. Source agent fields agree with the released harness for these95 matches;
no harness-mismatch claim is supported.

Reproduction: scripts/audit_cheatbench_upstream_identity.py, report
terminalbench_original/UPSTREAM_IDENTITY_V2.json. Earlier exploratory
UPSTREAM_REPLAY.json used a dictionary keyed only by trial name and mixed row
counts with unique names; it is superseded for count interpretation. Immutable
raw downloads and both reports are retained. No neural monitor has run, and no
safe/violation labels have been independently certified by these source checks.
The released card explicitly includes unauditable cases in benign; localization
provenance naming a model prediction is not proof of independent human labeling.
