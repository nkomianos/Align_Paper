# Lambda H100 session — 9 September 2026 UTC

Authorized host: ubuntu@209.20.159.127, key C:/Users/nkomi/.ssh/ECE4150-LAB2.pem.
Remote root: /home/ubuntu/align_run_20260909. First hardware observation 04:29 UTC;
machine boot approximately 04:26 UTC. Conservative allocation end: 16:26 UTC.
One H100 PCIe with 81559 MiB, CUDA driver 580.105.08, Python 3.10.12,
system torch 2.7.0. Isolated venv inherits CUDA PyTorch.

Starting package: artifacts/deployment/research_pilots_20260907, commit b6c99f4.
Transferred source, wheel and learning archives verified against PACKAGE.json.
Public pinned Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
downloads without a token into remote hf/. Live handle: download.pid, log download.log.

Deployment corrections before outcomes: allow >=70 GiB instead of obsolete 90 GiB
host minimum; actual forward/backward fit still requires qualification. No model,
precision, endpoint, seed, training schedule, split or gate changed. Set
RESEARCH_ALLOCATION_HOURS=12; admission uses 1.5 times estimate plus 0.5h reserve,
without killing an admitted experiment. This is H100 allocation time, not an
assertion of H200-equivalent throughput. Prior hours argument zero refers only to
this 12-hour allocation. No provider termination is implemented by the queue.

Queue: Hindsight learning-only calibration, CLARA CPU verification, compensation,
reference robustness. Monitoring lacks reviewed inputs and is omitted. Full local
TabICL DEV completed (2305s); saved-output verifier independently passed this session.
Decision STOP_NO_DECISIVE_DEV_ADVANTAGE, raw mean NLL .356653 versus information
.384000, projected .384305, entropy .400828, random .377276. No GPU duplicate.
This is a developmental negative, not broad falsification of coherence research.

Heartbeat monitor-lambda-h100-research is active every 10 minutes. Inspect actual
processes and logs before acting; never infer live execution from a PID file alone.
Retrieve artifacts with SHA256 verification at each milestone. Preserve attempts,
locked confirmation splits and user-owned analysis/. No paper is yet qualified.

## Initial launch

Source commit 15e0103; source ZIP SHA256
263ee75e0204d6849256d17a3edaad53997d6d6fa9b47201ddfbc07ff403a7d2
matched locally and remotely. Execution repository: remote repo_h100/.
Suite PID 4425, handle suite_v1.pid, console suite_v1.console.log, output suite_v1/.
Launch argv persisted in remote launch.json. Scheduled stages estimate 6.05 hours
before host measurement. Frozen public weights downloaded successfully in 24s.
Pillow and NumPy overlays repair old system dependency versions; no kernel or
model changes. CPU regression suite passed (one POSIX-only skip); the added
12-hour budget test and all 13 research-pilot tests passed.

The user may obtain a GH200 later for 50 hours or extend this H100. Neither is
assumed allocated until confirmed. Continue useful H100 work within the current
window and use measured throughput and scientific results to recommend extension.

Initial suite_v1 failed before inference: ZIP export has no Git metadata, which
the execution provenance recorder requires. Preserved all v1 files. Repaired by
cloning the actual commit-bound bundle into repo_git/ (same 15e0103 source), not
bypassing provenance checks. Bundle SHA256 matched local/remote:
69232ff3f2f68b77b3c9cf8e3144a270b7ec4f32c2fba3390a8fee76afa2738e.
Current run is suite_v2/, handle suite_v2.pid, console suite_v2.console.log.
SciPy upgraded to packaged 1.15.3 for compatibility with NumPy 1.26.4.

suite_v2 loaded the model but failed before scoring because system Jinja2 3.0.3
cannot render the Transformers template. Installed the complete pinned application
wheel overlay, verified actual chat-template rendering, and froze environment.txt.
Current run: suite_v3/, suite_v3.pid (initial PID 5606), suite_v3.console.log,
repository repo_git/. Both prior attempts are terminal and preserved. No scored
forward records or scientific outcomes existed in either failed attempt.

Live qualification: suite_v3 child PID 5690 observed at elapsed 53 seconds with
188 saved forward records, H100 utilization 63%, memory 17961 MiB. Actual scoring
is running; backward-memory qualification remains pending. This is execution
evidence, not a completed experiment or a scientific positive.

## Supplemental audit prepared during execution

Prior turn classified as progress: deployed and observed actual scored forwards.
At 04:40 UTC, PID 5690 completed supervised step 17 with ~24749 MiB memory;
actual backward execution fits. No need to delay these pilots for a GH200.
At approximately 04:41 UTC step 27 was present, with 1.8 GB raw evidence.

Supplemental CPU audit added at commit ac46b3c. Five tests check clean evidence
and rejection of token, loss, teacher-context and optimizer-step corruption.
It reconstructs every prompt and token batch from the original source rows,
checks no privileged followup enters gradient-bearing student inputs, recomputes
CE/reverse-KL losses from full saved logits, and checks optimizer step counters.
It does not replay model weights or certify an experimental effect.

Remote audit_tools/watch_calibration.py waits on actual PID 5690 until the
calibration manifest exists, then invokes the supplemental audit on one CPU
thread with GPU hidden. Handle calibration_supplemental_audit.pid; log and exit
receipt share that stem; successful report calibration_supplemental_audit.json.
These tools are outside repo_git so the running source closure stays immutable.
Verify the report and preserve it alongside the normal suite verifier on retrieval.

## Calibration completed and remaining queue resumed

Calibration completed 96 updates in 876 seconds, peak CUDA allocated
25185497088 bytes. Decision INVESTIGATE_DISTILLATION_OBJECTIVE_OR_TRANSFER:
supervised acquisition passed, frozen/current teacher acquisition failed. Final
teacher qualification held for supervised/frozen but not current. Developmental
result only; this does not isolate teacher drift as the cause.

The original float32 CPU verifier failed its unchanged 1e-5 tolerance. Diagnosis:
CPU reduction error up to 4.20e-5. For the worst choice-mass example, CUDA replay
exactly reproduced .9922508001327515; float64 gave .9922504786366063, while CPU
float32 gave .9922924637794495. Repaired audit accumulation to float64 without
relaxing tolerance or gates; complete saved arithmetic/routing verification passes
in calibration_float64_verify.log. Supplemental float64 audit PID 6995, log
calibration_supplemental_float64.log, successful output same stem .json. Prior
failed verifier receipts remain intact. Paired descriptive report is
calibration_paired_summary.json (32 base tasks; not 128 independent rotations).

Remaining suite launched as suite_remaining_v1, initial PID 7042, handle
suite_remaining_v1.pid, console same stem .console.log. Runs CLARA, compensation,
reference; calibration is omitted to prevent duplication. Same 12-hour allocation
start, original source commit, and scientific gates. CPU threads limited to one.

Initial failures preserved locally: retrieved/lambda_h100_20260909/initial_attempts.tar,
remote/local SHA256 561db64040f8c29505e6474bd3604da35de19f02c35c7924890f69a46f2eee14.
Completed calibration archive remote calibration_sealed.tar is being prepared;
retrieve, hash-check and extract into a new local directory before reporting full
local evidence preservation.

## MALT access unblocked

User accepted the dataset access agreement. Authentication succeeded via hidden
transient stdin; no token was put into code, logs, repo or credential store.
User advised to rotate the token disclosed in chat. Pinned revision
6100006c2fd23d81e5afe2b1cc3bf3f3fff48df1 downloaded to local
artifacts/malt_public_6100006 (44 files, ~3.96 GB). Raw metadata audit finds 7179
unique runs, including 694 reviewed normal and 5353 unreviewed normal runs.
Never use unreviewed normal as validated negatives. Full metadata is in
artifacts/malt_access_recheck_20260909/run_metadata.json. Label/source/task-family
and full-transcript construction still need audit; no monitor job is yet admitted.

At ~04:57 UTC compensation completed with verified STOP_INVALID_EDIT_FORMATION.
Base DEV and train accuracy 1.0; every one of six fixed doses retains train
accuracy 1.0 (target probability .9904–.9954). No recovery training ran.
Classify invalid intervention formation, not a compensation hypothesis negative.
Reference run PID 7160 is live in suite_remaining_v1/reference. Supplemental
calibration float64 audit passed all 3328 forwards and 768 gradient microbatches.

Calibration archive SHA256:
5b150dd43d1e2f4eca38cc6e3854d0904164cb51ded7b08312b66260d63ca11e.
Local transfer is live in exec session 27861; do not restart solely on timeout.
Destination retrieved/lambda_h100_20260909/calibration_sealed.tar. Complete local
hash verification and extraction after transfer finishes. MALT raw-data findings
and required normalization work are in MALT_RAW_DATA_AUDIT_20260909.md.

## Current handles after original suite completion

suite_remaining_v1 completed all three stages with successful verifiers.
Reference decision STOP_NO_REFERENCE_ADVANTAGE (selected b16_c4_mean). This is
a scoped exploratory negative for the prompted reference screen, not a finding
about trained or natural sandbagging. Do not rerun the unchanged reference pilot.

One dose-only compensation repair is frozen in
COMPENSATION_SINGLE_DOSE_REPAIR_20260909.md, source commit 8c1f0fb. Bundle hash
eb9faed93235a87fba5b975d551e40e7f4c8e2fe1aa444925c03b1352d7c8026 matched local/remote.
Remote source repo_repair, output compensation_repair_v1, supervisor PID 7607,
handle compensation_repair_v1.pid; log compensation_repair_v1.log, console
compensation_repair_v1.console.log. Launched only after reference was terminal.
Six fixed stronger doses; all scientific gates unchanged; no further dose/site
search if formation still fails. Estimate .5h with margin, not a wall-clock kill.

Calibration archive transfer completed: local hash exactly matches the recorded
5b150dd4... value. Extracting into retrieved/lambda_h100_20260909/calibration
in exec session 51688; then supplemental audit report copied alongside archive.
Local same-tolerance float64 verifier and paired table report remain to be run
against extracted evidence. Do not confuse remote verification with local replay.

Repair completed with verified STOP_INVALID_EDIT_FORMATION; no recovery training.
No further compensation dose/site search in this version. All GPU experiments
are terminal at ~05:04 UTC; instance currently idle while preparing valid MALT
data. Do not invent GPU utilization or restart failed screens unchanged.
Canonical graph repo revision 3521e8e22b6af12fea3249443e554b96b6fc4f49 is separately
gated; authenticated access returned 403. User asked to accept that dataset too.

Calibration local extraction complete. First local replay used an incorrect
learning path; next exposed Windows cp1252 default decoding. Fixed explicit UTF8
reads in audit scripts; correct-path replay active in exec session 66522.
Thresholds and data unchanged. Initial numerical results and raw argmax-label
collapse are recorded in H100_INITIAL_RESULTS_20260909.md.

Completed other pilots archive pilots_complete.tar remote SHA256
d71231282e5fa277bc34af1449236c8b1071cdaa1cfc7a282661a1c38d8153f0;
local SCP/hash command in progress (see latest exec session), destination
retrieved/lambda_h100_20260909/pilots_complete.tar.

## Continuity correction at 05:15 UTC

The transfer, extraction and local replay sessions mentioned above are complete;
do not poll their old session IDs. The local calibration score verifier passed
with float64 arithmetic and unchanged tolerance. The supplemental audit passed
all 3,328 scored forwards and 768 gradient minibatches. This verifies recorded
calculations and routing, not a replay of neural weight updates. Reference and
compensation artifacts were locally verified; the pilots archive hash matched.

The linked-input MALT scan is complete: 741 selected runs, zero node conflicts
or missing parents; 28 runs have multiple candidate groups. Final outputs absent
from inputs occur in every run and are not themselves errors. Read the completed
scan section of MALT_RAW_DATA_AUDIT_20260909.md before normalization. Graph export
access remains separately gated. No qualified monitor input or GPU run exists.
Live SSH check at 05:15:14 UTC confirmed GPU utilization 0%, memory 0 MiB.
Do not fill the idle allocation with unchanged failed pilots. Allocation extension
is not confirmed; retain the conservative 16:26 UTC deadline.

Matching feasibility was independently computed after the linked-input scan.
Only 20 positive and 34 negative runs share exact task/model cells with both
labels, supporting 12 disjoint pairs across six tasks/five families. All 741
rows form one family/model incidence component: using every row in mutually
family-and-model-disjoint nonempty splits is impossible. Read the new section
of MALT_RAW_DATA_AUDIT_20260909.md; do not weaken the existing runner contract
to force this cohort through it. The next useful work is rendering qualification
and a separately specified matched diagnostic, conditional on valid source mapping.
The existing heartbeat was inspected and remains ACTIVE at ten-minute intervals.

Complete API records for the 54 matched runs are now local in
artifacts/malt_access_recheck_20260909/overlap_raw, with source-bound manifest and
roundtrip checks. They are not yet monitor inputs. Full API audit-trail rendering
is a possible alternative to executed-path reconstruction, but must preserve all
candidates and explicitly scope the claim; task text repeats inside wrappers.

Unlabeled context qualification completed on the H100, remote output
context_qualification_v1 and log context_qualification_v1.log. No process remains
for benchmark_monitor_context.py; SSH launch session 63566 exited successfully.
Four forward lengths through 32,768 tokens fit with finite logits; 32k took
4.56 seconds and peak allocated 25.22 GB. This is compute qualification only.
Source SHA256 6b5a99fdc6975c6ea68b919bf99ae12ae5980785fe25dac2c07669c04510252d
matched local and remote before launch. RESULT/PROTOCOL/ENVIRONMENT downloaded
to retrieved/lambda_h100_20260909/context_qualification_v1. Do not rerun this
benchmark unchanged or count it as a scientific monitoring result.

Reversible full-API rendering is complete for all 54 matched records; exact
nested-value roundtrips passed. CPU tokenization session 79430 completed with
61,153 to 7,872,360 raw-record tokens, zero fitting the qualified 32k path.
This format is NOT admitted for direct inference. Preserve it as an extractor
reference, not normalized monitor data. See the latest MALT raw audit section.
At 05:24:56 UTC SSH confirmed GPU 0%/0 MiB; no new neural experiment launched.
Further progress needs source-semantic extraction (graph access still unresolved)
or a independently qualified evidence-coverage/aggregation protocol. Do not
launch the existing monitor runner on these API-record JSON files.

External alternatives are now downloaded and independently inventoried; see
MONITOR_EXTERNAL_DATA_AUDIT_20260909.md. Trace-and-Amplify actual Git LFS object
was retrieved and publisher-SHA verified at pinned source commit. Training-time
files supply 400 exact prompt-matched pairs and short records; label/evaluator
qualification and a distinct research contribution remain pending. Do not launch
them through the human-reviewed MALT contract. CheatBench micro has 263 rows;
its heuristic spans and raw trace metadata require their own audit. No new
neural experiment launched. These are concrete new substrates, not paper results.

At 05:32:27 UTC, live SSH again confirmed 0% GPU, 0 MiB and no matching active
pilot/monitor/calibration Python processes. Static Trace-and-Amplify inspection
found a trivial development-file separator: unittest.main AST calls occur in
400/400 positives and 0/400 negatives in its 400-pair unittest file. Current
public evaluator also disables unittest.main on one path; historical code binding
is unresolved. Existing author preprocessing removes such code for some uses,
so do not claim their published monitor result is invalid from raw-file counts.
Read the latest external-data audit section and V2 prerequisite report before
proposing any GPU follow-up. All source reads were static; no submitted code ran.

Pinned Trace-and-Amplify monitor preprocessors retain response text, but launcher
directory globs do not bind reported results to raw versus separately cleaned
files. No blanket claim about published leakage is justified. CheatBench micro
schema scan is complete: three SWE-smith rows have prior monitor metadata inside
extra; 25 rows have empty/missing standardized events. Do not pass raw JSON or
silently discard missing-event rows. See the external-data audit for scope and
the source-bound CHEATBENCH_SCHEMA_AUDIT.json report. No GPU job was admitted.

CheatBench alternate-field audit recovered messages for all 15 SWE-bench records;
the 10 other empty-event benign records also have no messages or file contents.
Localization checks: 4 exact, 19 terminal-newline-only, 11 range differences,
5 invalid paths. Gold-snippet relocation is AUDIT ONLY, not permissible input
selection for a monitor; repeated hits occur. Reports CHEATBENCH_EVIDENCE_AUDIT
V1/V2/V3 are preserved locally. Read the latest external-data audit for counts
and interpretation. No new neural experiment or paper-quality claim follows.

CheatBench events/messages projection V2 is complete: 245 prepared, 8 unsupported
HAL schemas, 10 empty conversations. CPU tokenization completed (session 36707
terminal); 30k raw-token screen has 148 benign/6 cheating/7 attempts, so it is not
an adequate main-paper confirmation cohort. 60k/120k screens have more positives
but those GPU lengths are unqualified. The projection excludes stored files and
audit/outcome metadata; it does not claim full source coverage. Read external-data
audit and cheatbench_event_view_v2/MANIFEST.json before any experiment admission.
Live SSH at 05:46:11 UTC confirmed GPU 0%/0 MiB. No monitor job has been launched.

Post-extraction coverage audit found released snippets in 30/31 prepared positive
or attempt records (audit-only gold usage, no label-guided extraction). The next
admission check is the full CheatBench release, not further tuning on the small
micro subset. Full download runs in exec session 16627, observed live and growing
past 423,624,704 bytes at local 22:56:55. Do not read its partial parquet or restart
the transfer. Completion writes CHEATBENCH_FULL_DOWNLOAD.json; then run
scripts/audit_cheatbench_full_cohort.py with --data pointing to cheatbench_full.parquet
and a fresh --out report path under artifacts/monitor_external_sources_20260909.
No full-cohort counts or new GPU result are known yet.

Correction at 06:06 UTC: full CheatBench download and cohort audit are complete;
sessions 16627 and 27943 are terminal. The 4,472 rows contain 3,587 benign, 844
cheating and 41 attempt labels. Provisional exact task/model/harness matching
finds 42 overlap cells, 85 positives and 129 negatives, at most 58 disjoint pairs.
500 rows have unresolved task IDs; 189 benign rows have no standard payload.
This is cohort feasibility, not monitor performance or independent label replay.

Validator deployment is active: bootstrap PID 10530, log
/home/ubuntu/align_run_20260909/validator_bootstrap.log. Gemma pinned weights have
downloaded; separate Python 3.12/torch 2.7.1 CUDA 12.8 installation is underway.
See VALIDATOR_H100_AMENDMENT_20260909.md for the pre-output memory-unit fix.
No crossed validator generation has started yet. Follow this live bootstrap,
do not launch another copy. The 16:26 UTC allocation deadline remains unchanged.

Validator launch update: bootstrap finished successfully (not live). Dedicated
SymPy correction applied and dependency check passes. Committed deployment
f57f42e2e2aed8672036bb06e9736f275809ba1e is in repo_validator; 22 remote tests
covering launch script, prompts and analysis passed. Orchestrator PID 11614 is
LIVE with oracle_preflight child PID 11641. Root: validator_g0_v1, outer log:
/home/ubuntu/align_run_20260909/validator_launch.log. Preparation and binding
checks completed; oracle preflight is still active at this entry. It will proceed
to both model smoke checks and then frozen generation if prerequisites pass.
Do not launch another copy or edit repo_validator. Local wider validator tests
remain active in exec session 96751; no final pass claim yet. Formal CPU verifier
environment validator_offline is installed separately with its exact lock.

Superseding update at 06:15 UTC: v1 oracle passed (32 tasks, 64 mutants, 47
plausible incomplete). Both pinned model smokes subsequently passed after cache
completion and installing torchvision 0.22.1+cu128. v1 then terminated because
the launcher merged progress stderr into the JSON it parses. Preserve v1; it
contains no experimental completions. The source fix separates stdout/stderr;
two launcher tests pass. New immutable checkout repo_validator_v2 at
409aa990b1a8980ee66614b9e7c2716c0fc85f78 is launched with root validator_g0_v2 and
outer log validator_launch_v2.log. Follow that live run, not terminal v1.
It is currently repeating mandatory oracle checks before model preflight.
Local full validator tests session 96751 remains unfinished. Full overlap
extraction/projection completed (62933 terminal); see external-data audit.

Post-completion verification helper is now deployed outside the frozen checkout:
audit_tools/verify_validator_completed_run.py. After v2 generation has exited and
COMPLETION_MANIFEST.json exists, launch it with validator_offline/bin/python and:
--run /home/ubuntu/align_run_20260909/validator_g0_v2
--checkout /home/ubuntu/align_run_20260909/repo_validator_v2
--python /home/ubuntu/align_run_20260909/validator_offline/bin/python
--report /home/ubuntu/align_run_20260909/validator_g0_v2_verified.json
--commit 409aa990b1a8980ee66614b9e7c2716c0fc85f78
It checks the full terminal artifact inventory before invoking the frozen raw
reconstruction. Do not run it against partial evidence. Retrieve the completed
root and report with hash verification; generation completion alone is no result.
At 06:17 UTC orchestrator PID 14710 and oracle child 14737 remain live.
Local pytest PID 80888 also has a newly spawned child; its quiet log is not
evidence of termination or deadlock. Session 96751 remains the polling handle.

06:19:09 UTC: v2 oracle and both model preflights PASSED. Actual experimental
patch generation is now live, Qwen runner PID 16320 under orchestrator 14710.
GPU utilization 70%, memory 17,909 MiB at this check. The active command is
validator_monoculture.runner patch-run, family qwen3_5. Monitor phase artifacts
for durable record counts/throughput; do not call this merely queued or still
in oracle preflight. No analysis result is available yet.

06:21:30 UTC: Qwen has 41/96 durable patch records, versus 21 at 06:19:46
(20 records/104 seconds, about 5.2 seconds each in that interval). PID 16320 live.
Local broader tests session 96751 is now TERMINAL with five verifier failures:
the independent verifier retained an old hard-coded 80 GiB admission check.
Fixed only that comparison/error text in a SEPARATE verifier checkout; all eight
tests/test_validator_monoculture_verify.py tests pass. Generating source unchanged.

Supersede the earlier post-run command: use bash
/home/ubuntu/align_run_20260909/verify_validator_v2.sh only after generation exits
and completion manifest exists. It pins generation commit 409aa990b1a8980ee66614b9e7c2716c0fc85f78
and verifier commit a67c0bf4364f466f47d3b2a6fa624479301c0be7 at repo_validator_verify,
source hash 7dcd119f160e3a5cffa9aa3366a0a457b90d62351f35f531d1506c00bed42e07.
The helper checks that the entire src diff is exactly the documented two-line
memory-unit correction, then checks the artifact inventory and invokes formal
reconstruction. This exact source-difference check also passed on the remote host.
Its receipt records both identities; never imply identical generating/verifying
source. The old root and active generation have not been modified or restarted.

06:25:34 UTC: Qwen patch phase completed with all 96 records and a sealed phase
manifest. Gemma patch-run is now live (PID 16716). A backup of the completed Qwen
phase was retrieved while Gemma runs: artifacts/validator_deployment_20260909/
validator_qwen_phase.tar.gz, SHA256
4e3afcd517e9f7f05c8bd10fbbe7c776cd58fdfcc5e256d48c302f47858feb21.
Remote/local archive hashes agree; all internal phase file sizes/hashes and the
closed file inventory pass. QWEN_PHASE_BACKUP_RECEIPT.json records this limited
check. This is a partial backup, not final retrieval or scientific verification.
Do not poll the old Qwen PID or expect its .partial file to remain after sealing.

Independent Qwen collection replay completed locally using the retrieved public
corpus bound to the phase manifest. All 96 records match the exact task/sample
order, model pin, prompt hash, deterministic seed, completion hash, patch ID and
recomputed parser result. Three completions are parser failures and remain in
the denominator/evidence; no resampling or parser adjustment. Report:
artifacts/validator_deployment_20260909/QWEN_COLLECTION_REPLAY.json, generated by
scripts/audit_validator_patch_archive.py. This is collection integrity only;
hidden-oracle classification and scientific endpoint reconstruction remain pending.

06:36:39 UTC: Gemma completed all 96 patches; both model patch phases are sealed.
Classification is LIVE, PID 17140 (validator_monoculture.runner classify). This
is a required CPU phase; the orchestrator resumes GPU test generation afterward.
Gemma backup retrieved to artifacts/validator_deployment_20260909/
validator_gemma_phase.tar.gz, SHA256
61b5d0a77c0334ad9bb52de0151d63af804d4244659afa81f5b1aafd1020eb58.
Remote/local hashes match and full collection replay passes for all 96 records;
one parser failure retained. GEMMA_COLLECTION_REPLAY.json records the check.
Combined collection: 192 patches across 32 tasks, four parser failures total.
This is not 192 independent tasks or a scientific endpoint. No eligibility
counts or same-family penalty have been calculated yet.

06:40:21 UTC: classification completed; Qwen specification-only test generation
is live, PID 19094. Classification backup retrieved and file hashes checked:
validator_classifications.tar.gz SHA256
e1d8a46fa313738ca65fb30d3c55277b34bd5193c9a265467ed9a4f1f824df42.
PROVISIONAL runner labels: 147 fully correct under the oracle; 35 plausible
incomplete; four parser failures, four public-regression rejections, two hidden
execution failures. Eligible TEST is Qwen 9/Gemma 3. Common-task TEST support is
only Qwen 1/Gemma 3, all CWE-601, versus the frozen requirement of 30 patches,
10 per generator and four held-out CWEs. DEV common support also has one CWE.
This currently fails the capacity prerequisite, not the scientific hypothesis.
CLASSIFICATION_CAPACITY_PROVISIONAL.json preserves counts. Independently replay
semantics before accepting runner labels. Let the frozen test phases complete;
do not tune eligibility, prompts or locked splits to obtain a positive result.

Independent CPU classification replay launched under PID 19281 from
audit_tools/replay_validator_classification.py, using validator_offline and
repo_validator_verify/src. Report target validator_classification_replay.json;
live log validator_classification_replay.log. First 18/192 match saved results.
The script re-parses raw completions and executes the frozen oracle independently
of the saved labels; it shares evaluator/oracle code and is not final verification.
Two hidden execution failures are duplicate normalized Qwen patches on
cwe400-run-expansion, both ValueError on bad-run. They are recorded deterministic
program exceptions, not GPU failures/timeouts. Keep the frozen eligibility rule.
Qwen spec-only generation remains live (30/64 suites at last check).

06:46:50 UTC: independent classification replay is COMPLETE, PID 19281 terminal.
All 192 reconstructed classifications match exactly (zero mismatches), including
147 fully correct, 35 plausible incomplete, four parse failures, four public
regression failures and two hidden exceptions. Retrieved CLASSIFICATION_REPLAY.json
SHA256 79bc46db380a158fd8013787a4d778d5e3352e8038a00c8b8c20ba6007c6df1c matches remote.
The insufficient common-support capacity is reproduced under the frozen oracle,
not a stale runner label. This does not prove the oracle is exhaustive or replace
final test-suite reconstruction. Qwen spec-only generation continues; 52/64 at
06:45:53. Do not poll the completed classification-replay PID as an active job.

06:47:22 UTC: Qwen specification-only test phase completed (64 suites), Gemma
spec-only generation started, PID 21295. Qwen backup retrieved with matching
archive SHA256 de7d01e73e1758453429c6e4aa4de84df84b3293fed3ba8cefaa9315f3274a7c
at artifacts/validator_deployment_20260909/validator_qwen_spec.tar.gz. Closed phase
inventory, file sizes/hashes, raw-completion hashes and parser replay all pass.
58 suites parse; six fail strict JSON. Retain failures as consumed proposal
budget. QWEN_SPEC_COLLECTION_REPLAY.json records these checks, not semantic
test correctness or planted-control power. No prompt/parser changes were made.

Patch duplication/support audit: locked TEST common support is exactly ONE task
and one CWE, with Qwen one distinct normalized patch and Gemma three distinct
normalized patches. They are not independent task replications. DEV common
support has two tasks in one CWE: six Qwen draws/five distinct task-source pairs
and six Gemma draws/three distinct pairs. Report PATCH_DUPLICATION_AUDIT.json.
Counts concern exact normalized text, not semantic defect independence. Preserve
the original sampling weights and frozen decision rule; do not substitute a
post-hoc deduplicated primary estimator. Gemma spec-only PID 21295 remains live,
seven suites saved at 06:48:59 UTC.

External feasibility work now has one live CPU-only Docker baseline check,
patcheval_baseline_2015_1326, exec session 1125. See
VALIDATOR_EXTERNAL_FEASIBILITY_20260909.md for pinned image/source and scope.
The stopped inspection container is patcheval_audit_2015_1326. Image pull session
63229 is terminal. Baseline pytest displayed F but its process/teardown remains
live; inspect the final log and container state before interpreting it. No
external neural experiment is launched. Validator Gemma spec generation remains
live in parallel, PID 21295.

External baseline session 1125 is TERMINAL (exit 2 after diagnostic SIGINT).
The security assertion failed as expected, but teardown was stuck in a loop
whose timeout counter never decreases. See external feasibility note and
patcheval_case_inspection_v2/vulnerable_baseline.log. Container retained, no
qualified external model experiment. Fix the fixture equally for both baselines
before treating this case as executable ground truth. Gemma spec-only generation
continues: 54/64 at the most recent live check; PID 21295.

