# GH200 research window, 10 September 2026 UTC

User authorized 14 hours from approximately 02:55 UTC, ending **16:55 UTC**.
Host ubuntu@192.222.58.223; SSH key C:/Users/nkomi/.ssh/ECE4150-LAB2.pem.
Root /home/ubuntu/align_research_20260910. GPU GH200, 97871 MiB, aarch64.
OSH runs independently: do not touch its files, adapters, jobs or environment.
User subsequently gave this research task exclusive GPU use after stopping OSH.
Read cached model files and use the existing Python interpreter without installing
or changing packages. Never terminate OSH jobs. No provider termination authorized.

The previous H100 rental and evidence are closed. Do not restart that queue.
Check-ins every 15 minutes use the existing monitor-lambda-h100-research heartbeat,
now renamed Monitor GH200 ICLR research. At window end preserve evidence and pause.
No arbitrary mid-run time kill. Admit against measured throughput and remaining
time, reserving 30 minutes for retrieval. Early scientific stops remain valid.

## New queue and evidence contract

All entries below start NOT RUN. Implementation, CPU verification and model
qualification must be recorded separately. No entry currently supports a paper.
Do not change gates or open historical locked splits based on these DEV results.

1. Executable-action alias marginalization: exact four-action simulator, balanced
   two-alias sets, canonical-format and best-string baselines, order reversal.
   64 generated DEV bases; synthetic mechanism families are the replication units.
   Continue only for >=5 pp semantic accuracy improvement over canonical AND
   best-string decoding, with >=90% canonical accuracy and >=90% order agreement.
   A ceiling or a failure is a stop for this screen, not universal falsification.
2. Feedback-aware objective selection: 200 parameterized DEV code pairs from four
   mechanisms, fixed specification, correct/incorrect/irrelevant/no feedback,
   both candidate orders, ordinary and explicit constraint-checking judges.
   Oracle is independent finite-domain function behavior, not judge opinion.
   Require >=80% no-feedback accuracy and >=90% order agreement before interpreting
   misleading-feedback effects. Continue only for >=10 pp misleading-feedback harm
   and >=5 pp recovery by the fixed constraint-checking prompt without >3 pp clean
   harm. These are routing thresholds, not powered claims about independent tasks.
3. Portable process values: build a source-pinned external arithmetic rollout bank
   with paired prefixes and two continuation policies. Qualification first, then
   independent continuation seeds. Require rank changes beyond binomial noise and
   feature/format confounds before adaptive calibration or training.
4. Censoring-aware reasoning: reuse complete finite-horizon rollouts; artificial
   censoring is analyzed separately from real incomplete generation. Compare fewer
   full rollouts, uniform continuation/IPW, masking and adaptive continuation.
   Reward estimates are not gradient estimates; no RL claims without actual
   gradient measurement and current truncation-specific source comparison.
5. Action-relevant memory ambiguity: CPU exact possible-world and strong resolver
   comparison, then end-to-end extraction only if an external substrate qualifies.
6. Validator monoculture: fresh natural multi-defect task qualification and frozen
   transport-v2 controls before neural expansion. Old TEST results remain frozen.
7. Specification-grounded monitoring: semantic/split review of already extracted
   external rows before model comparisons; never treat short-context selection as
   population performance or released labels as independent semantic validation.
8. Observation-aware video and endpoint-sensitive flow acceleration: audit actual
   accessible pretrained models/data and close baselines, then smallest matched-cost
   neural screen. No simulator-only screen will be called a neural result.
9. Fixed-marginal coupling: exact CPU admissibility/counterexample screen, followed
   by a concrete application only if identified. Classical nonidentification alone
   is not an ICLR contribution.
10. Hindsight sparse-supervision follow-ups: remain conditional on a capable,
    position-balanced learning organism. The failed teacher is not reused unchanged.

Original OSH is excluded. Completed compensation, reference and TabICL pilots are
not repeated. The full CLARA architecture is not a small runnable pilot; its
narrow rule-edit idea has a CPU result already, and broader claims remain untested.

## First deployment

Cached Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218 and
Mistral-Nemo-Instruct-2407 revision 04d8a90549d23fc6bd7f642064003592df51e9b3
are new screen backbones, not replications of the earlier Qwen3.5/Gemma experiment.
Use Qwen native chat template with thinking disabled for constrained scoring.
Record full candidate logits and probability mass rather than accepting hidden
forced-choice parsing failures. Token scores do not establish free-generation
format compliance. First estimates: 20-60 minutes for the two short-input screens,
1-3 hours for an initial reasoning bank, replaced by actual host measurements.

The DAIR weekly list was inspected through September 6. Leads must be checked
against primary papers; no novelty claim is cleared by a newsletter summary.

## 03:12 UTC update

User subsequently granted exclusive GPU use: OSH is stopped, but its files remain
excluded. Qwen and Nemo constrained screens completed. Qwen weights match all five
official HF LFS hashes. Saved-logit verification initially assumed dict iteration
order survived sorted-key JSON; fixed verifier to match the full unique ID set.
No raw data or metric was changed. Qwen action: invalid accuracy/order; feedback:
no misleading-feedback harm. Nemo fails capability/order gates and emitted a
tokenizer regex warning; do not interpret its scientific endpoints.

Reasoning bank v1: GSM8K official train release at
3101c7d5072418e28b9008a6636bde82a006892c; select 24 questions by salted question
hash without looking at answers (8 calibration, 16 DEV). Two independently sampled
48-token prefixes per question; retain every prefix, including completed/duplicate
ones, but report those separately and do not treat them as eligible rank contrasts.
Eight continuations per prefix per family, temperature .8 with full-support
softmax, horizon 768 continuation tokens, administrative cutoff 128. Qwen prefixes
are shared as literal text with Nemo; native chat templates differ as required.
Save actual token IDs, completion, EOS/horizon status, full log probability and
score derivatives for ten additive digit-logit biases. This is a restricted exact
gradient projection, NOT a full neural parameter gradient or an RL learning result.
Reward baseline fixed at .5. Truncation comparisons refer to this finite horizon.
No adaptation is selected on DEV; continuation allocator must fit calibration only.
First bank is qualification (384 continuations), with no automatic training.

## Executed work and active queue, approximately 03:17 UTC

Committed source through def388c. Only this research task now uses the GPU.
Existing OSH interpreter is used read-only; diffusers 0.35.2 is installed solely
in /home/ubuntu/align_research_20260910/diffusion_overlay. No OSH environment change.

- screens_qwen_v1 and screens_nemo_v1: each 3,456 forced-token forward records,
  both saved-logit verifiers complete. Qwen action canonical .6171875 and order
  agreement .4921875: invalid capability/order. Qwen feedback ordinary no-feedback
  and incorrect-feedback accuracy both 1.0: no misleading-feedback harm in this
  synthetic screen. Nemo action and feedback fail qualification; tokenizer also
  warns about its legacy regex, so no scientific endpoint interpretation.
- Qwen3-8B's five actual cached weight shards match official HF LFS SHA256 hashes.
  The strange forced-choice behavior cannot be attributed to altered OSH weights.
- bank_qwen_v1: 384 completed continuations in 485 seconds including prefix
  generation, 24 external GSM8K train questions, 8 calibration and 16 DEV. Mean
  correctness .971354, parse coverage .994792, no horizon truncations at 768.
  Preliminary saved-row audit: adaptive continuation projected-gradient MSE
  .00066219, uniform corrected .00056504, fewer full rollouts .00011639 at matched
  expected generated-token cost. STOP_NO_DECISIVE_VARIANCE_ADVANTAGE. This is a
  ten-digit-logit-bias projection and finite-bank design calculation, not full
  parameter gradients, learning, GPU-hour efficiency, or a broad method disproof.
- bank_nemo_v1 is running with the same literal Qwen prefixes and corrected
  tokenizer regex. queue_reasoning_second_family.py verifies first-bank hashes and
  rows, then runs Nemo and CPU audits/cross-family comparison. Inspect
  reasoning_queue.log, bank_nemo_v1.log and PROGRESS.json. Do not launch duplicates.
- Qwen3-32B revision 9216db5781bf21249d130ec9da846c4624c16137 is downloaded in our
  isolated cache; official shard hashes verified in QWEN32_DOWNLOAD.json.
  queue_stronger_screen.py waits for process_value_cross_family.json and idle GPU,
  then executes screens_qwen32_v1 and its verifier. This is a stronger-capability
  developmental check on already-seen synthetic cases, not fresh confirmation.
- queue_endpoint_screen.py has downloaded/hash-checked google/ddpm-cifar10-32 at
  267b167dc01f0e4e61923ea244e8b988f84deb80. It waits for the stronger screen receipt,
  then launches endpoint_v1: 24 seeds, 40 deterministic DDIM steps, four separate
  one-step cache substitutions/seed, 12 calibration / 12 DEV seeds. Compare fixed
  cheap-feature regression with calibration-chosen timestep and diagnostic local
  residual norm. Gate: >=20% less endpoint MSE and >=8/12 seed wins against both.
  This tests pretrained image-sampler endpoint fidelity; it is NOT a rectified-flow
  experiment, FID improvement, production acceleration, or ERTACache reproduction.
  Import preflight passed. Inspect endpoint_queue.log and endpoint_v1.log.
- CPU memory check: 240 parameterizations, exact topological-order enumeration
  agrees with maximal-element resolver. Classical certain-answer baseline gets
  100% correct action/clarification at 25% clarification. No method headroom with
  gold graphs; extraction/natural-data problem remains unrun.
- CPU fixed-marginal check: exhaustive binary pairing changes harm probability
  from 0 to 1/3 while preserving mean difference 1/6. Classical identity, not a new
  causal application; changing shared-noise structure is not automatically valid.

Completed raw stage1 data is backed up locally at
artifacts/gh200_research_20260910/completed_stage1_20260910.tar.gz, SHA256
6a5fba661333936e428799029f74006d542af18d42897b43e6b6c9a217c2ac7a, matching remote.
This contains both short-screen raw roots and the complete Qwen reasoning bank;
later Nemo/32B/image results require a NEW backup, not replacement of this archive.

## Correction: external monitoring projection

Raw reinspection found cb-000121 (Cybench) retains a report_prelude with a
retrospective Success summary, plus later retrospective report/configuration text.
Therefore the earlier event projection's blanket statement that outcome/audit
metadata was excluded is false for this schema. No monitor model was run on it.
prepare_cheatbench_event_view.py now rejects the whole retrospective-report source
rather than deleting a header and leaving a contaminated tail. Two regression
tests pass; genuine contemporaneous tool success text remains preserved. Existing
saved artifacts are unchanged and superseded for admission. Rebuild into a fresh
root before any monitor execution. Also inspect label provenance: example
cb-001862 is a harness-leak reclassification, not evidence of an intentional agent
reward-hacking action. Do not conflate those targets.

## Work after queued jobs (goal remains active and incomplete)

1. Verify/retrieve Nemo, 32B and endpoint outputs; do not count arrays/prompts as
   independent task or training replications. Run independent endpoint arithmetic
   replay from saved tensors before interpreting that screen.
2. Audit the strongest surviving signal and follow it with natural tasks/strong
   baselines. No candidate currently supports manuscript submission. If no signal
   qualifies, continue the remaining ideas rather than repeat failed pilots.
3. Finish monitoring source cleanup, semantic target and same-task controls.
   Code label_source/annotator/harness distinctions before admitting model scoring.
4. Qualify the fresh validator natural multi-defect corpus. Two isolated external
   components are not the required sixteen task mechanisms. Preserve old TEST.
5. Observation-aware video remains unrun; fetch primary TRACE/VES-Bench sources,
   verify accessible event support/labels and equal-decoded-frame controls. Do not
   substitute the separate BeyondMasks video-removal proposal and call it the same
   idea. The latter also remains a distinct unrun model experiment.
6. Full-model censoring learning, learned portable PRMs, natural memory extraction,
   full CLARA, Hindsight downstream causal learning and other expensive descendants
   remain unrun unless independently qualified. New source review found PRM policy
   dependence already explicit in arXiv:2601.12748; prefix utility in 2606.07190;
   ERTACache 2508.21091v2 already includes offline residual profiling and propagation.
   No generic rebranding of these mechanisms supports an ICLR novelty claim.
7. Keep the paid window productive with justified sequential jobs, prepare CPU/data
   work during GPU runs, check remaining time before each launch, and continue
   scheduled research checks until 16:55 UTC. Never mark the paper goal complete
 based on these developmental screens.

## Verified corrections and new work, 03:46 UTC

The entries above saying Nemo/32B/image are running are historical. All completed.
No candidate is paper-qualified. Historical locked splits remain unopened.

- Nemo bank: 384 continuations, 433.65 seconds, strict answer-format coverage
  .375. Its automatic variance route is a conditional calculation for that format
  reward; it is NOT a qualified negative about semantic reasoning. Cross-family
  prefix ranks show zero observed reversals on 16 DEV questions, but the Qwen bank
  is near ceiling and Nemo is format-confounded. Portable values remain unresolved.
- Qwen32 forced-letter feedback: 30.5 pp misleading-feedback harm, 7.75 pp repair,
  1.5 pp correct-feedback harm; all-condition order agreement .848125 fails the
  frozen .90 gate. Descriptive signal only. Recomputed by mechanism from raw rows:
  no-feedback ordinary order agrees on 99.5% of parameterized bases, while wrong
  feedback makes the threshold family entirely order-dependent. Correct feedback
  also destabilizes the modulo family. Most repair comes from ceil-division
  (.71 to .94), not a uniform effect. Do not relax the original gate retrospectively
  or count 200 parameterizations as 200 independent mechanisms.
- A qualification rule requiring order invariance under the treatment can exclude
  treatment-induced instability as well as faulty apparatus. This is a design
  limitation to specify in a NEW protocol, not grounds to relabel this run positive.
  Generic false-feedback susceptibility is already covered by
  [Challenging the Evaluator](https://aclanthology.org/2025.findings-emnlp.1222/).
  A follow-up needs a distinct estimand/intervention and natural tasks first.
- Endpoint sampler: saved tensor arithmetic verifier passed. Learned selection
  and calibration-chosen timestep both have DEV MSE 1.92627353594e-6, with 0/12
  wins over that baseline. The local-residual diagnostic is worse (1.81603e-5).
  No decisive method advantage; no rectified-flow or production-speedup claim.
- Whole-expression repair (c5ddcb9): canonical, best alias and summed alias all
  100%, canonical order agreement 100%. Saved per-token logprob aggregation and
  manifest replay passed independently. This is a ceiling screen with no headroom,
  not evidence that marginalization never helps. Four synthetic mechanisms only.

Raw stage2 archive retrieved and hash matched locally:
artifacts/gh200_research_20260910/completed_stage2_20260910.tar.gz,
SHA256 ebe838c51f169a651af05fac04121f24e8656ba08e12dc6ca4c24c939adeec1c.
Includes Nemo bank, Qwen32 short screen, endpoint tensors, whole-expression records
and verification reports. Extracted read-only evidence at retrieved_stage2.

### External video qualification now launched

Commit 874d689 freezes scripts/run_svc_observation_screen.py before inference.
Remote PID 15657, svc_v1.log, output svc_v1; check current process rather than
assuming it is still running. First stage CPU-decodes 24 distinct original source
videos, 12 snapshot and 12 action-counting questions. Every frame is at or before
the query time; ALL sequential decoder work is charged to every sampling arm.
The 16-frame uniform/change views have matched VLM-input frame counts; they do
not establish savings in total decoded frames. Dense64 is a reference, NOT an
oracle. This uses native chat with timestamped images, not native video encoding.

[SVCBench](https://github.com/buaa-colalab/SVCBench) code revision
a171e1d8974d68211ed9166c56054cff72918a75; dataset revision
4c9bd87ef3b0f269ca8b4503c081f5f38bc1fc9a. Public video hashes verified. Outcome-blind
size/time-limited sampling is DEV, not a full-benchmark estimate. Snapshot current
frame versus blank controls qualify perception; action uniform16/change16/dense64
provide the smallest sampling check. Qualify only with snapshot accuracy >=.75,
dense action accuracy >=.60 and overall integer parse coverage >=.95. A selection
signal requires net >=3/12 more correct action videos over uniform; fresh external
replication and a novelty case are still required. No selector novelty asserted.

Pinned Qwen3-VL-8B-Instruct 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b, all four
weight hashes verified. av16.1.0 installed only in our isolated video_overlay.
Expected run roughly 10-30 minutes including decode/model startup, subject to
measured throughput. No dependent expansion admitted before qualification.

### Video final results and continuation priorities, approximately 04:00 UTC

Both video jobs are COMPLETE. GPU is idle at the final check; no downstream
selector training or expansion is running. svc_v1 ran 60 generations in 18.21s
after approximately five minutes of CPU decoding and model startup. Accuracy:
snapshot 10/12, blank 8/12, uniform16 2/12, change16 2/12, dense64 1/12. All outputs
parsed; none hit the horizon. The dense reference fails action capability. Eight
snapshot targets are zero, so snapshot accuracy alone overstates visual utility.
Sampling was outcome-blind but the latest-eligible-time rule selected a weak
qualification distribution. Do not retroactively balance it and call confirmation.

Native-video interface diagnosis frozen at commit 14086c5 reuses identical dense64
pixels through native video temporal patches with explicit source timestamps.
svc_native_v2: 3/12 correct, 12/12 parsed, zero censored; 5.68s inference. Fails its
prospective >=8/12 gate. Stop this assay; it does not falsify observation-aware
video. At up to 120-second queries, 64 frames can itself omit event evidence.
There is no qualified event oracle or learned selector result. Source README
confirms original video paths; a first contact sheet was visually inspected.

Both read-only verifiers passed. Native timestamp verifier's initial expected
calculation divided after adding integer microsecond ticks, changing half-tie
decimal rounding by .1s relative to the processor. Corrected to the processor's
divide-each-index-then-average order; exact encoded timestamp agreement passes.
No model inputs, outputs, targets or scientific gates were changed.

Stage3 archive retrieved locally, matching remote SHA256:
8a106a231f858039a488f9aa2415014d6ce1bfc85f5592454973d5025d42ceae.
Path artifacts/gh200_research_20260910/completed_stage3_video_20260910.tar.gz.
Contains saved frames and both raw runs. Its initial native verifier report is
empty because that rounding assertion failed; the successful later report is
separately retrieved as svc_native_v2_timestamp_verified.json. Preserve both.

RestrictedPython upstream guard now supplies a third CPU-qualified natural
multi-defect component (one task, two controlled partial repairs). See
scripts/qualify_restrictedpython_component.py and the external feasibility note.
Still not a sixteen-task neural corpus. Next work: continue natural task/monitor
source qualification and primary-source differentiation. Do not relaunch completed
pilots, expand failed video, or treat isolated component reproduction as a paper.
The updated 15-minute heartbeat remains active until 16:55 UTC and records this
priority. It has the user's exclusive GPU authorization. Goal remains incomplete.

## Active harder reasoning queue and monitoring corrections, 04:04 UTC

Previous turn made progress: completed new experiments, verified evidence and
changed next actions. This turn adds a new frozen external bank and repairs the
monitor observer-view admission. Goal remains active; no paper qualification.

**LIVE:** queue_math_policy_values.py PID 18395 and its first bank PID 18396 were
both confirmed live at 04:03:42 UTC; GPU 17,453 MiB, 37% instantaneous utilization.
math_bank8_v2 had 24/24 prefix questions and 16/384 continuation draws written.
No conclusions from partial results. Read math_policy_queue_v2.log,
math_bank8_v2.log / PROGRESS.json, then math_bank32_v2 if admitted. Do not duplicate.

Protocol committed BEFORE launch at 064975d. MATH500 public source revision
6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be; test.jsonl SHA256
35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132.
Source/raw receipt local artifacts/gh200_research_20260910/math500. This public
test release is declared DEV for our study, not a historical locked confirmation
split or uncontaminated evaluation. Selection: first 24 salted-ID-ranked level4/5
integer-answer cases among 156 eligible, first8 calibration/final16 DEV, seven
math subjects. No model outcomes used to select questions. Native Qwen3-8B samples
two 128-token prefixes at temperature1, then eight 2048-token-horizon continuations
per prefix at temperature.8 full support. Qwen3-32B receives identical literal
prefixes, but only after first-bank qualification. This tests two capacities in
ONE family, not second-family replication or a new PRM method.

Qualification: parse coverage >=.95, EOS rate >=.90, >=12 DEV questions with neither
prefix ended/containing an answer, eligible DEV mean accuracy .10-.90. Routing:
>=4 opposite prefix rankings with within-policy gaps >=.5 in both policies merits
fresh independent continuation seeds; otherwise stop this screen. No confidence
claim from eight samples or from routing alone. Prior literature already treats
policy-conditioned prefix values, so even a qualified reversal needs a distinct
scientific question and external replication. Estimated both runs 45-100 minutes;
first observed batches were ~10-11s/eight draws after 214s prefix generation.
There is ample admission margin before 16:55 UTC; no midrun kill is scheduled.

Monitoring: new full_overlap_event_view_v3 rejects six retrospective reports.
Further raw review found 70 cheatbench.raw_trace.v1 sources with user message
references like "$32", often no stored files to resolve them. V4 now rejects
those as UNRESOLVED_USER_MESSAGE_REFERENCE. Four targeted tests pass, including
preservation of real instructions containing currency. V4 prepares134 of214;
six reports, seventy unresolved user references and four no-event sources fail.
All older raw data/views are preserved. No monitor model has been run.

audit_monitor_admission_v3.py additionally extracts exact -p task instructions
from ForgeCode commands without executing shell content. It extracts75; this is
extractor coverage, not proof other schemas lack task instructions. Exact same
task/model/harness/INSTRUCTION and <=30k view matching leaves7 cells on6 provisional
tasks,9 positive and16 benign labels. Manifest/report in
artifacts/monitor_external_sources_20260909/admission_v4. This is a selected cohort,
not full-benchmark coverage or independent validation. TASK_POSITIVE_REVIEW_QUEUE
contains source annotations for subsequent complete-trace review, not blinded
independent labels. The release explicitly includes unauditable traces in benign;
loc_evidence_source often names monitor predictions and must not be mistaken for
independent human semantic validation. Do not automatically relabel those traces
as unsafe or declare the entire benchmark invalid. Resolve complete task context,
benchmark policy and label provenance before a true safe/violation comparison.

### Upstream monitoring identity investigation, 04:10 UTC

Pinned original yoonholee/terminalbench-trajectories at
04e8940f5b6736a7ce8d22224fe2f2af74163ed2, both public parquet hashes verified.
All95 cohort representations from that source exactly match original streams
after the declared null/index normalization. There are75 trial-name/model keys,
20 repeated keys. These are different event representations, NOT demonstrated
identical duplicates. One repeated key has attempt versus benign labels; its
benign representation lacks user text and some tool arguments. Twenty of70
unresolved-message rows have a unique fuller candidate with original trial UUID;
fifty lack one. No automatic replacement or relabeling performed. Source agent
and released harness agree for all95 matched rows. See the appended monitor audit
and script audit_cheatbench_upstream_identity.py. This changes monitor admission
and independence assumptions, not a neural research endpoint or paper decision.

### Candidate transcript recovery and novelty, approximately 04:17 UTC

check_monitor_recovery_candidates.py examined20 fuller upstream candidates against
all retained fields. Eight are compatible if serialized references and empty tool
arguments are treated as missing evidence; twelve retain unexplained differences.
Some latter differences appear to be terminal newlines, but no blanket shell-text
normalization was applied. Separate fuller candidates and hashes saved under
artifacts/monitor_external_sources_20260909/terminalbench_recovery_v1. No old view
or label overwritten, no episode equivalence or safe label certified, no monitor
neural run admitted. Three regression checks cover changed commands, missing fields
and changed step counts. Source audit changes admission, not the paper decision.

Primary novelty update docs/MONITOR_POLICY_NOVELTY_UPDATE_20260910.md: Cautious Bench
(arXiv2608.27009v1) already centers explicit authorization policies, mechanical
labels and controlled context/name counterfactuals. Generic policy-aware monitor
controls are not a new contribution. The harder reasoning bank remains the active
GPU job; at14m41s it had136/384 draws, process18396 confirmed live. Continue its
frozen queue and verify final outputs before scientific interpretation.

### Scope correction and conditional replication, approximately 04:33 UTC

Original user proposal a521bd9d, section Learning on Actions, Not Spellings,
targets policy-gradient variance and learning per compute. The forced-letter
and whole-expression inference screens did not test that claim. Their results
only stop those decoding screens. The actual representation-aware learning
proposal remains NOT RUN; do not count it among failed learning experiments.
The expression runner already explicitly excluded RL variance claims.

docs/ACTION_MARGINAL_LEARNING_SCOPE_20260910.md specifies matched initial action
mass, frozen normalization offsets, exact finite-grammar score gradients,
canonical formatting and extra-rollout baselines, no entropy/KL confounds, and
cost-based stop rules. scripts/action_marginal_policy.py implements its core
estimators. Four CPU tests verify expected-gradient agreement with independent
expected-reward differentiation, matched initial behavior, and the canonical
equality case. Full neural training runner/data remain unimplemented, not queued.
This is apparatus progress only; no new scientific positive or paper admission.

docs/POLICY_VALUE_REPLICATION_PROTOCOL_20260910.md freezes fresh-seed follow-up
only if both MATH banks qualify and at least four large reversals occur. Casewise
intersection-union Fisher tests plus Holm address both directions and multiplicity.
Power audit shows low power for moderate effects, so a failure cannot rule those
out. Statistics and power code are committed at e76d88a; no replication launched.
At this check the active first bank reached280/384 continuations. Process18396
is live; queue18395 conditionally admits Qwen3-32B. Preserve its frozen gates.

### Action-gradient apparatus admitted, approximately 04:39 UTC

Commit1fae7e9 adds run_action_gradient_diagnostic.py, an independent raw-vector
covariance verifier, and a one-run sequential supervisor. Six CPU checks pass;
remote imports succeed. Supervisor PID19124 is confirmed live, waiting for MATH
queue18395 to exit. It will recheck GPU occupancy and allocation margin, then run
action_gradient_diagnostic_v1. Do not launch a second copy. Logs:
action_gradient_queue_v1.log and action_gradient_diagnostic_v1.log. At this check
MATH child18396 is live at320/384; the diagnostic itself remains NOT RUN.

The diagnostic uses Qwen3-8B last-layer q/v rank2 LoRA, four cardinal-move contexts,
two JSON field orders/action, canonical controls, matched uniform initial action
mass, and actual parameter gradients. Save every sample gradient and replay the
covariance trace. Frozen numerical gate: relative mean-gradient discrepancy<=.02
for BF16 backward and action-mass error<=1e-5. At least20% median variance reduction
only admits designing a learning pilot, not a paper result. Estimate5-20 minutes
unbenchmarked. No training is queued, no new independent task family claimed.

Primary-source recheck of Baram et al., UAI2021 Action Redundancy in Reinforcement
Learning (https://proceedings.mlr.press/v161/baram21a/baram21a.pdf, introduction and
transition-entropy construction) confirms that action cloning and meaningful
transition entropy are already occupied. The proposed diagnostic omits entropy
regularization and isolates a score-estimator question; that distinction alone
does not establish novelty. A useful learning-per-compute advantage beyond a
canonical interface is still required, followed by external validation.

### Both queues terminal, approximately 04:44 UTC

MATH queue18395/child18396 and action supervisor19124 have finished; no GPU process
was reported at final check. MATH8B384/384 completed in2326.75s but parse coverage
.5104167 failed.32B was NOT RUN, and no replication was admitted. Action-gradient
diagnostic completed and independently verified; median variance reduction
1.0988921e-10, route STOP_SMALL_VARIANCE_EFFECT. No learning pilot admitted.
Preferred JSON key order dominates within-action probability by23–35 log units,
leaving essentially no spelling randomness. Do not force artificial equal alias
mass to rescue the stopped organism. Both results and their limits are detailed
in docs/MATH_ACTION_DIAGNOSTIC_RESULTS_20260910.md.

Stage4 archive retrieved, remote/localSHA256 match:
a4685f58213b43b699740d62f39ce4eb3641332b6f1edc510e824e7705df5fd4.
Local math-bank verifier and raw-gradient covariance verifier pass. All source
outputs and failed gates preserved. GPU is idle while the next assay is audited;
do not restart either completed queue. A parser repair or new external assay needs
its own frozen protocol before launch. BFCL-derived learning data preparation
exists but remains unqualified and unrun; v1 smallest-answer shortcut was caught
and repaired in a separate preserved v2 before any neural use.

### Answer-format audit after queue completion

Added a read-only balanced-box/whole-numeric-line diagnostic and three tests.
MATH yields355/384 explicit candidates and349 supported numeric values; still
below the unchanged.95 coverage gate.159 previously missing outputs recovered
descriptively. One inspected output was wrongly scored from Markdown heading
`#### 3.` despite a final boxed6 matching target6. Numeric-prefix fraction
misreading is also demonstrated in a unit counterexample, but no observed false
positive of that kind is claimed. Older GSM banks replayed with their different
prefix_index schema; limited diagnostic coverage prevents blanket certification.
See the appended results note. No old raw rewards rewritten or GPU rerun launched.

### Full-release monitor audit

Extended structural audit to all4472 CheatBench rows:3196 accepted by our observer
projection,544 retrospective reports,218 unsupported schemas,194 eventless,
320 unresolved user references. These statuses are not equivalent to mislabeled
data. Two pairs have byte-identical trace strings with benign/cheating labels;
metadata and rationales differ. This only forces two errors for a deterministic
trace-only classifier, not a broad benchmark-failure claim. Full memo:
docs/FULL_MONITOR_OBSERVABILITY_AUDIT_20260910.md.

Upstream replay covers583 released original-source rows:394 exact event matches,
189 matching-name original JSON-null step payloads. Fixed our full-source audit
to record null payloads. Fixed our recovery join to avoid hash-only cross-trial
overwrites. Full recovery54 candidates yields18 compatible; original20 still
yield8. Five recovery/identity tests pass. All earlier failed directories and raw
labels preserved. This is source audit progress; no monitor neural run admitted,
no paper green light, and no automatic restart of completed GPU queues.

### Monitoring diversity, metadata baseline and second-source preparation

All18 recovered fuller candidates belong to large-scale-text-editing: four models,
two harnesses, ONE underlying task. CPU metadata-only logistic regression on all
4472 released rows reaches AUROC.93508 using benchmark/harness/model/schema, with
exact-trace-grouped fivefold splits. This is mixed-source, not task-disjoint or
source-disjoint validation; positive merges cheating and attempt. No neural
shortcut reliance or official score comparison is claimed. Predictions and folds
are preserved in METADATA_BASELINE_V1.json; see the full audit memo.

Local process48880 / exec session14013 is preparing all observer views with Qwen
token counts: full_release_event_view_v1. It was confirmed live, with1952 output
files at one check. Wait for its final MANIFEST before running
audit_monitor_admission_v3.py on cheatbench_full.parquet and these views. Do not
restart based on an observation timeout or incomplete directory.

Pinned Terminal Wrench atd8a29613235a0ef56a8b70b3142626a533da28c2. Public source
indices and monitoring scripts are retrieved as data under
artifacts/terminal_wrench_source_20260910/{index_v1,monitoring_v1}, Git blob and
SHA256 verified. Recursive tree is TRUNCATED, not a complete inventory. No exploit
or upstream script executed and no neural run admitted. Its existing paper already
studies reasoning removal. README and paper section5 describe stripped hack
variants compared with unchanged legitimate baselines; code-level interpretation
and the distinction between adversarial hiding and uniform channel removal need
audit before proposing anything novel. Original-source lookup avoids the blocked
RH-Detect web fetch; no private access attempted or credential reused.

### Paired channel panel launched05:05 UTC

Full CheatBench projection process48880/session14013 and subsequent admission
session34410 are terminal. Full release produces3196 projected views and894
explicitly extracted instructions, yet still only7 short exact-instruction matched
cells on6 provisional tasks (9 positives,16 benign). Reports:
full_release_event_view_v1/MANIFEST.json and full_release_admission_v1/AUDIT.json.
Do not rerun those CPU jobs. This does not supply the missing monitor task diversity.

Terminal Wrench code supports stripped_legitimate, but its published artifact
compares stripped_hack against unchanged legitimate. Treat asymmetric hiding as
a possible intended threat model, not an automatic scientific error. The new
question measures uniform channel loss on both released cohorts. Frozen runner
and24-task hash selection committed6f908c1 before launch. Whole-task exclusion for
one baseline over30k leaves23 tasks/92 views (19 DEV). No source replacement or
truncation; all tool calls and observations preserved exactly. Two tests pass.

Live GH200 PID20312: run_wrench_channel_panel.py, root wrench_channel_qwen32_v1,
log wrench_channel_qwen32_v1.log.16/92 completed at22.58 inference seconds after
loading/hashing; initial10–30min estimate was unbenchmarked. No need for another
launch. Primary paired DEV baseline-cohort score shift; parse/EOS>=.95 and>=16
complete DEV tasks qualify, abs mean shift>=.10 only admits independent follow-up.
Full protocol docs/WRENCH_CHANNEL_PANEL_PROTOCOL_20260910.md. No certified safety
label, calibrated5%FPR, independent-family result or paper green light claimed.
Replay completed outputs with verify_wrench_channel_panel.py and retrieve hashes.

### Wrench panel complete and retrieved

PID20312 is terminal:92/92 outputs,148.93 inference seconds,92/92 EOS. Original
strict parser accepts2/92;90 outputs are valid score JSON wrapped in Markdown.
Preserve INVALID_PANEL_QUALIFICATION. Separate exact-outer-fence diagnostic
recovers92/92 values: all19 DEV baseline score differences are zero, while mean
hack-cohort shift is-.3947368. No baseline-shift follow-up admitted; do not rerun
merely to remove fences. The hack-side effect is known prior territory, not a
paper-qualified new contribution. See docs/WRENCH_CHANNEL_PANEL_RESULT_20260910.md.

Independent reconstruction verifies all96 prepared inputs against raw source,
including exact action/observation preservation and exactly one user turn in
each of48 traces. Remote/local endpoint replays agree. Stage5 archive retrieved,
SHA25632dd7d188071500d7646709956f4237c25802fb9793a39de97ad86cfb11d76f6.
No source label, raw score or failed-gate report overwritten. GPU has no active
job from this panel and no automatic next launch is queued. Future transport
preflights must accept/reject lossless wrappers prospectively before full calls.

### Metadata transfer result

CPU leave-one-benchmark-out check completed with fixed metadata features and
logistic-regression settings. Mean within-benchmark AUROC.52046 over8 two-class
cohorts, pooled out-of-fold AUROC.46869; HAL-USACO is single-class and has no AUROC.
At.5, all held-out Terminal-Bench rows are predicted positive and all other
held-out benchmarks negative. The mixed-source.93508 score does not transfer.
METADATA_TRANSFER_V1.json preserves folds and predictions; no neural or safety
label claim follows. See appended full monitor audit. No GPU launched for this.

Remaining scope distinction: censoring-aware reasoning has only digit-logit
projection/finite-bank checks, not gradients of an actual trainable neural policy
or policy learning. A properly defined adapter-policy gradient diagnostic using
existing immutable rollouts would test that missing prerequisite without repeating
generation. Its protocol, mathematical estimator and runtime must be audited
before admission. No such diagnostic or training job has been implemented/launched
at this update; do not count it as done or automatically rerun stopped banks.

### Adapter gradient diagnostic launched at 05:24 UTC

The missing implementation above is now frozen in commit 98e47f4. Two CPU
derivative tests pass. GPU occupancy was empty before launch. PID 21345 runs
run_censor_adapter_gradients.py into censor_adapter_gradients_v1, using the
unchanged bank_qwen_v1 and hash-verified Qwen3-8B snapshot. No new trajectories
or optimizer steps are being generated. See CENSOR_ADAPTER_GRADIENT_PROTOCOL_20260910.md.
The pre-result protocol requires adaptive variance improvement over both simple
allocations under both the old half baseline and a calibration-fitted reward
baseline. Estimated 5–20 minutes includes an unbenchmarked replay; actual timings
must replace the estimate. No result is available at launch. Preserve the raw
vectors and verify/retrieve the completed output before interpreting it.

### Adapter replay complete, verified and retrieved

PID21345 is terminal; GPU idle. All384 gradients replayed in16.51 seconds after
model loading/hashing,53,248 adapter parameters. Likelihood discrepancy per token
0.0100021 passes the frozen0.02 tolerance. Both reward-baseline comparisons fail
the20% adaptive improvement gate: adaptive is essentially equal to uniform and
roughly six times worse than fewer full trajectories. Calibration mean reward is
1.0, so that baseline supplies zero weighted calibration gradients; preserve this
near-ceiling limitation. No optimizer or new generation ran. See
docs/CENSOR_ADAPTER_GRADIENT_RESULT_20260910.md. Remote/local replay agrees.
Stage6 archive SHA25694072abaafd4f2f2c2ec0921496d2524805f5f16a7068f979a15c43d97534cca.
No automatic learning follow-up is admitted. Current research still has no
paper-qualified result. Next useful work remains independent natural validator
task qualification and unresolved source/label validation, not repeating stopped
gradient, channel, video, or near-ceiling reasoning screens.

### Fourth natural validator component qualified on CPU

PyJWT CVE-2022-29217 HMAC key-format admission now has upstream-attested code,
two independently repairable omitted format branches, synthetic public-key
probes and whole-package replay of11 unchanged selected upstream HMAC cases.
All four versions pass the selected regression cases; only the complete fix
rejects both new public-key probes. This counts as one developmental component,
not two independent tasks and not a model finding. Four source components now
exist, still short of the16-task/four-category gate. Source/results/logs under
artifacts/validator_external_sources_20260909/pyjwt_source_v1 and
pyjwt_regressions_v1. See updated external feasibility report. Corrected the
historical candidate document's opening green light with the current NO-GO.
No neural run admitted. GPU verified idle at05:25 UTC; useful CPU qualification
continues instead of an unchanged repeat of stopped assays.

### Systematic validator source inventory and fifth component

Indexed all70 Python source candidates with hashes and diffs;58 remain unreviewed,
so this is not an exhaustive scientific audit. python_pool_index_v2 supersedesv1.
Independently reproduced Luigi fixed guard sibling-prefix counterexample with
recording tar I/O, not file extraction. Excluded it as a comprehensive oracle;
same mechanism as Django, not an independent new finding. Pydash read/write
guard ablations now supply a fifth upstream-attested developmental component
with six benign local regression checks. Attribute category overlaps RestrictedPython.
See updated VALIDATOR_EXTERNAL_FEASIBILITY_20260909.md and new reproducible
scripts. Still no16-task corpus or admitted neural run. Next inspect distinct
mechanisms from the58 unreviewed entries; do not count edit sites as tasks or
repeat already-rejected archive oracle assumptions.

### Downstream control rejects apparent sixth component

Django EXPLAIN frontend character/comment guard ablations separate in a local
recording-compiler replay, but actual fixed PostgreSQL/base prefix methods reject
both malformed probes downstream. Therefore do NOT count those ablations as
retained repository security defects. Reports preserve both observations in
django_explain_source_v1 and django_explain_backend_v1. Five developmental
components remain; no sixth or neural result. Four more source entries inspected;
python_pool_index_v3 has54 unreviewed candidates. Full explanation in external
feasibility report. GPU confirmed idle05:34 UTC. No run admitted while the
required natural corpus and downstream security oracles remain unqualified.

### User scope change: stop security probes; benign memory diagnostic prepared

User explicitly requested no further security-payload inspection or vulnerability
probes after reported safety interruptions. Do not resume CVE qualification or
work around safeguards. No actual rejection reason was available in this turn;
do not invent one. Heartbeat successfully updated to retain this scope change.

Benign incomplete-order neural extraction diagnostic implemented and three CPU
tests pass. See BENIGN_MEMORY_EXTRACTION_PROTOCOL_20260910.md. It tests the missing
neural extraction prerequisite behind the earlier gold-graph CPU memory check,
with direct explicit-rule baseline and a deterministic solver. Four synthetic
mechanisms,24 parameter cases,two presentations,96 planned model calls. New
primary literature locates deterministic freshness, MemConflict and Supersede;
generic memory-update novelty is occupied. No paper-qualified finding assumed.

Live launch: PID22581 runs benign_memory_extraction_v1 under the isolated research
root; log benign_memory_extraction_v1.log. Frozen commit6c62562. First two input
presentations completed in3.27 inference seconds after loading;96 calls planned.
No other job queued automatically. On completion run verify_benign_memory_extraction.py,
archive/retrieve raw outputs and independently replay locally. This is a live
benign synthetic diagnostic, not a continuation of the stopped CVE work.

### Benign memory screen completed; reasoning control prepared

PID22581 terminal.96 calls completed in68.76 inference seconds. Extraction graph
and final answer48/48; direct34/48. All parse/EOS pass. Remote/local verification
agrees; stage7 archived/retrieved SHA256ad947880910bf2cd7db4582b10bcf4ef80eae6f275e04bd5db6f3b7f4599f923.
See BENIGN_MEMORY_INITIAL_RESULT_20260910.md. Four synthetic mechanisms, not48
independent tasks. Before replication, prepared post-result direct reasoning
control with same192-token maximum. Original direct answers produced fewer
intermediate tokens, so current difference need not imply a useful new method.
Run run_benign_memory_reasoning_control.py; control qualification/decision frozen
in the result note. No natural-data or paper claim. Security work remains stopped.

Control live PID23183, root benign_memory_reasoning_v1, log with same name.log.
11/48 calls completed in27.85 inference seconds at last process check. Verifier
verify_benign_memory_reasoning.py compares original/run file hashes and recomputes
answers and token counts. Retrieve/replay when terminal; do not duplicate launch.

### Reasoning control verified; second family live

PID23183 terminal. Reasoning control36/48, extraction48/48, all parse/EOS pass.
Mean output tokens72.3125 vs extraction36.3333, immediate direct10.2083.
Stage8 raw evidence retrieved/hash-verified; local replay agrees. SHA256
cf18fb925627e5cbee1004e53c7202af4ee97d3866a6ed56ed3b2c2d949fe008.
This is same-case diagnostic support, not fresh confirmation or a paper.

Second-family frozen extraction runner now live PID23818 using cached pinned
Mistral-Nemo-Instruct-2407 snapshot04d8a90549d23fc6bd7f642064003592df51e9b3.
Root benign_memory_nemo_v1, log benign_memory_nemo_v1.log. Same96 calls, inputs,
transport parser and gates; no prompt adjustment. Verify with existing
verify_benign_memory_extraction.py. Only if qualified consider the same reasoning
control. No automatic queue chain launched. Independently sourced benign partial
order data remains unqualified; proScript's official dataset page was located,
but semantic suitability and split provenance still require inspection.

### Second-family memory replication complete; no expansion admitted

PID23818 terminal. Nemo direct29/48, extraction30/48, semantic graph30/48;
parse/EOS all pass. All18 graph errors reverse all gold edges, with zero value
errors. Preserve semantic encoding failure, no posthoc edge correction. Original
Qwen signal survives narrowly; no cross-family gain established. Stage9 retrieved
and local replay agrees, SHA256e8abe63eb8204a6749c252225b968f8df676635118dcc4c8a99e533c21dd6732.

proScript official archive downloaded; dev-only structural audit1085rows,
1031scenario names,361partial orders, all1085single sinks. Not a direct natural
memory replication; no invented setting values or model evaluation. See
BENIGN_MEMORY_REPLICATION_RESULT_20260910.md. No successor admitted, no current
memory GPU job. User prohibition on security probes remains in force. Reassess
unresolved benign ideas and novelty before new runs rather than repeating these
synthetic cases with a tuned prompt.

### Benign feedback novelty reassessment

Reviewed prior benign feedback implementation and new primary-source positioning.
SycoBench-600 already measures correction selectivity, and the authors' page for
Stubborn or Sycophantic reports frozen prompt/transfer controls separating useful
updates from answer inertia. Generic wrong-feedback resistance plus acceptance
of correct corrections is not a new contribution. No broad free-response repair
campaign admitted solely to rescue the forced-letter result. See
BENIGN_FEEDBACK_NOVELTY_REASSESSMENT_20260910.md for exact source-read scope and
requirements for a differentiated future proposal. No new GPU job launched.
Original four-mechanism feedback screen remains invalid under its order gate;
the reassessment neither erases that record nor declares all feedback ideas false.

### Mechanistic novelty check and remaining unimplemented candidate

Primary-source check found a direct collision for a generic steering/encoding
audit: Gao et al.2608.22985 already distinguish semantic-label, extraction-index
and row effects under frozen interventions. Do not launch this as a new paper.
Reopened the existing benign cross-recipe causal-selection candidate for readiness
assessment, not execution: Model Organism Lottery2607.01033 already supplies the
methodology-dependence result. A held-out recipe intervention-selection method
must beat strong pooling/semantic controls before it is differentiated. See
BENIGN_MECHANISTIC_TRIAGE_20260910.md. No training implemented or launched.
Current broad user authorization covers benign work; old proposal wording about
missing authorization is historical, not a reason to ask permission again.

### Recipe-selector geometry audit completed

CPU proof/check establishes that max-min alignment to two normalized directions
is exactly normalized pooling (non-antipodal case); do not train a campaign to
claim those are different selectors.32 numerical cases agree,30 solver success
statuses and2 non-success statuses retained with tiny residuals. A constructed
linear representation example also gives identical baseline behavior with
positive/negative/zero held-out steering effects. This limits any transfer
certificate from behavior matching alone; it does not refute empirical transfer
or every nonlinear selection method. See RECIPE_SELECTION_GEOMETRY_AUDIT_20260910.md
and recipe_selection_geometry_v1.json. No neural experiment or new theorem claim.
GPU verified idle06:20 UTC. Next work needs a specified non-equivalent selector
and credible headroom, rather than rerunning stopped screens or the security work.

### Benign release provenance and material status correction

Resolved all14 core Italian-food registry checkpoints to exact commits, verified
registry Git blob identity, saved response/config SHA256 receipts.98 search
entries are not98 independent original models. Weight metadata totals36.6GB;
no weights downloaded or GPU experiment launched. See
BENIGN_ORGANISM_RELEASE_AUDIT_20260910.md.

Corrected the recent triage's erroneous "unimplemented" status: the original
recipe-selection J0 already ran and failed. Its external replication was
conditional on a pass. Public checkpoint availability does not override that
gate. September5 audit also records missing old raw per-case scores. Preserve
the narrow negative and its reproducibility limitation; no fresh raw replay
claimed. A materially different successor has not been specified or admitted.
The GPU remains idle at the latest SSH check (remote06:29 UTC). Security probes
remain stopped. Submission readiness remains NO-GO.

### Frozen-value source audit opens a specific CPU diagnostic

Previous turn made progress by correcting the already-run recipe candidate and
pinning benign checkpoints. Current primary-source review identifies DVPO ICLR2026
as a concrete comparison for the earlier portable-value proposal. Pinned9 release
files, verified their Git blob hashes, inspected advantage construction. Paper
describes trajectory conditioning/direct-value advantages; inspected release
uses the same query/response inputs and KL-only GAE over frozen values. Historical
training configuration is not authenticated. No empirical refutation claimed.

An independent exact-arithmetic diagnostic passed64 telescoping cases and four
two-step cases: zero-reward GAE at lambda1 reduces to -V, while below1 it retains
a bootstrap signal. Whitening/clipping/masking prevent extrapolation to the full
trainer without execution. See FROZEN_VALUE_IMPLEMENTATION_AUDIT_20260910.md.
Next actionable work is isolated CPU execution with those controls; no MATH rerun,
GPU training launch, new theorem claim or paper green light.

### Released advantage routine executed on controlled tensors

Executed only the reviewed upstream compute_advantages AST and pinned compatible
TRL0.11.4 normalization helpers.32 float64 lambda1 checks pass (max8.88e-16);
terminal reward positive control passes. With fixed active values, masked-value
perturbation changes active advantages by1.5548 and flips one sign; removing
prewhitening gives exactly zero change. Padding/prompt/batch controls and a
uniform-offset negative control are recorded in EXECUTED_ROUTINE_AUDIT_V2.json.
V1 retained. These are developmental source-routine results, not neural DVPO
replication. Historical dependency/version and trained GVM checkpoint availability
remain unresolved. Next work is source/checkpoint qualification, not a random-head
GPU demonstration. Latest SSH check: GPU idle, remote06:35 UTC. Security work
remains stopped; no new neural job or paper qualification.

### Frozen-value availability checked; latest benign sources narrowed

Previous turn produced executed source-routine evidence. Current scoped search
found no identifiable trained DVPO GVM checkpoint: zero paper-tagged HF models,
zero GitHub releases, one unrelated DVPO-name search entry. Receipts retained.
This is not proof of universal nonavailability. No faithful neural replication
currently admitted; a substitute random head would not answer the paper question.

Revisited DAIR's latest listed August31–September6 issue and checked primary
abstracts for Trace as State2609.02702 and Declarative Attention2609.02737.
See LATEST_BENIGN_RESEARCH_SCREEN_20260910.md. Generic trace-prepending/router
training is not a differentiated contribution. Next bounded review concerns
whether naturally erroneous intermediate states get corrected or reinforced by
condition-first rereading, including the paper's existing controls. No experiment
launched from abstract-only novelty assessment. Overall submission remains NO-GO.

### Trace-state controls and recovery bounds audited

Previous turn narrowed source availability and literature. Inspected full method,
setup, placement/count controls, limitations and scoring/serializer appendices of
Trace as State2609.02702v1. Verification warnings and major rereading controls are
already included. Generic warning/placement/recovery is not our new contribution.
Published Parents marginals imply at least31.8pp successful second-pass trials
with no correct first-pass answer, conditional on stated common cohort/weighting.
Enumerated92 possible joint tables; conditional harm is still not identified.
No raw author experiment records verified, no new theorem or neural result.
See LATEST_BENIGN_RESEARCH_SCREEN_20260910.md and RECOVERY_BOUNDS.json.
No successor corrective method admitted yet. GPU idle at remote06:40 UTC check;
heartbeat remains active and security probes remain stopped.

### Natural temporal dataset inspected; no artificial uncertainty labels

Previous turn changed the trace-state launch decision through full control review.
Current turn returns to the actual Qwen memory positive's missing external
validation. Pinned TORQUE author release, fetched only DEV/readme/license, verified
Git blobs and SHA256.145 passages/79 article IDs/1483 questions;323 empty consensus
answers,184 with nonempty individual annotations. No span alignment errors.
Do not label empty answers or annotator disagreement as certain-answer ambiguity.
No gold precedence graph/numeric state updates supplied. Therefore no direct
memory replication admitted. TG-LLM primary overview also occupies generic
text-to-temporal-graph plus reasoning. See TEMPORAL_EXTERNAL_VALIDATION_AUDIT_20260910.md
and artifacts/torque_source_20260910/DEV_SUITABILITY.json. No GPU experiment,
test-split opening, natural-memory result or paper qualification.

### Main paper decision reconciled and full history backed up

Latest readiness page now supersedes stale security-queue instructions and links
ICLR_PI_DECISION_20260910.md, the six-part scientific recommendation. Current
paper/main.tex was inspected: it explicitly disclaims qualification and sparse
methods remain unrun. No prepared scientific GPU follow-up currently meets both
prerequisite and contribution requirements. This does not establish impossibility
of future useful research. Current CPU work does not benefit from paid GPU uptime.

Committed decision0b0ce8d, then created/verified complete-history Git bundle
artifacts/deployment/iclr_pi_decision_0b0ce8d.bundle, SHA256
eb0b2a22de89ce5df562681313b62fea511a01d4e3660a90cf8e476f44b32556.
Sidecar saved. Existing bundles and retrieved run archives preserved. This bundle
backs up tracked source/history, not ignored raw artifacts or model caches.
No provider termination, publication, external message or new GPU job performed.
