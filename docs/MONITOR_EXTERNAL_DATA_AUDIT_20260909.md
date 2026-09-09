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
