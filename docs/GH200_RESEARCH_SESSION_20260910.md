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
