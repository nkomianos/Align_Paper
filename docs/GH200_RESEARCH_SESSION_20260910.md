# GH200 research window, 10 September 2026 UTC

User authorized 14 hours from approximately 02:55 UTC, ending **16:55 UTC**.
Host ubuntu@192.222.58.223; SSH key C:/Users/nkomi/.ssh/ECE4150-LAB2.pem.
Root /home/ubuntu/align_research_20260910. GPU GH200, 97871 MiB, aarch64.
OSH runs independently: do not touch its files, adapters, jobs or environment.
User explicitly permits GPU use while idle; defer new launches if OSH is active.
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
