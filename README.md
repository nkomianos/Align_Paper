# AI Research Journal — ICLR 2027 exploration

This repository preserves the hypotheses, implementations, failed attempts,
measurements, and decisions from our ICLR 2027 research exploration. Its original
project was **Under Extinction**:

> Can an extinction-style intervention fingerprint identify what controls a language agent's actions when ordinary behavior is matched—and predict later specification gaming better than current behavior or knowledge probes?

It now also contains independent alignment, agent-security, and vision candidates.
No historical OSH adapter or old OSH paper artifact is imported or packaged.

## Scientific status

### CPU-only follow-up

The [matched Hindsight learning run](docs/HINDSIGHT_MATCHED_LEARNING_20260904.md)
completed six actual LoRA training arms (144 updates, six minutes CPU), with all
adapters/optimizers preserved. Copying feedback has worse NLL than an initially
matched fixed-noise control, but changed-wording accuracy is 8/16 for both.
Anchors yield9/16 versus8/16 anchor-only, with no unanchored accuracy improvement.
Even direct truthful supervision transfers poorly. This is a weak acquisition
regime and a diagnostic feedback effect, not a successful paper method.

The [released-template fresh preference test](docs/HINDSIGHT_PUBLISHED_TEMPLATE_CONTROL_20260904.md)
completed 128 CPU forwards: published hindsight 25/32, direct preference 27/32,
redacted wrapper 30/32. All three still favor option A; uninformative controls
choose A throughout. This qualifies useful but imperfect teacher signals, not
endogenous-feedback learning harm. Matched parameter learning is the next step.

The [paired history control](docs/HINDSIGHT_HISTORY_CONTROL_20260904.md) completed
96 more CPU forwards: shown prior answer 25/32, omitted 16/32 (all choices A),
redacted 30/32. Shown predictions replay exactly. This is a prompt-sensitive
apparatus finding, not full SDPO or a new cleanup method; the actual published
teacher construction remains a required control before learning experiments.

The [fresh Hindsight preference-revelation probe](docs/HINDSIGHT_REVELATION_PROBE_20260904.md)
completed 144 CPU forwards: explicit feedback selects the stated preference
56/64 times (25/32 when reversing the prior action). Endorsement does not
reinforce the prior action more than generic thanks (paired probability
difference −.05125). Complete scores were recovered after a summary serialization
failure; original evidence is unchanged. No learning or paper effect is claimed.
The [human-data provenance audit](docs/PUPPET_MESSAGE_PROVENANCE_20260904.md)
also found scripted first USER messages in every valid record and broad-topic
overlap across the proposed query split. That reconstruction extension is parked.

A [bounded-influence population check](docs/HINDSIGHT_PARTIAL_IDENTIFICATION_20260904.md)
constructs sharp binary preference intervals and a conservative update filter.
It retains useful updates in designed examples, but depends on a valid influence
bound; a tested misspecification counterexample shows compatibility alone is not
enough. This is not an SDPO result or paper greenlight. The next comparison must
include learning directly from the same sparse anchors.
That [finite-sample comparison](docs/HINDSIGHT_ANCHOR_VALUE_20260904.md) now
shows a narrow designed-model benefit over conservative anchor-only learning
when anchors are scarce, but not over ordinary anchor-only mean utility. The
benefit disappears at 256 anchors. Structural validity and observation cost,
not more GPU training, are the next unresolved questions.
The [stress test](docs/HINDSIGHT_ANCHOR_STRESS_20260904.md) now shows that allowing
large reporting-model error removes the gain, and sufficiently cheap extra
anchors outperform the combined rule. A realistic low-cost-log setting and
defensible measurement assumptions are prerequisites for an LM run.
A [human-data search](docs/HINDSIGHT_HUMAN_DATA_SEARCH_20260904.md) located a
public pre/post-belief dataset and completed a schema-only audit. Inclusion-rule
and data-license questions remain; no participant outcome analysis or training
has been performed.
The [reconstruction metric audit](docs/BELIEF_RECONSTRUCTION_METRIC_AUDIT_20260904.md)
adds a negative control: shared rating noise alone creates a .707 error/change
correlation for a perfect latent-belief estimator. Such a correlation must not
be treated as evidence of hindsight bias. Paired error metrics are implemented.
The [nested-prefix feasibility check](docs/BELIEF_RECONSTRUCTION_COHORT_20260904.md)
identifies 305 non-personalized conversations for a third-versus-sixth-user-turn
comparison that preserves early evidence. This is a new developmental cohort,
not the original study's filter or a model-performance result.
The [query-cluster precision check](docs/BELIEF_RECONSTRUCTION_PRECISION_20260904.md)
now assigns 72 conversations to DEV and 233 to confirmation without reading
endpoints. With hypothetical within-query correlation .3, confirmation has
about 47 independent-observation equivalents; this is a sensitivity calculation,
not measured power or a human-data effect. No inference has started.

The [ScopeJudge audit](docs/SCOPEJUDGE_OFFLINE_SCHEMA_AUDIT_20260904.md)
verified 4,897 labeled calls and constructed metadata-filtered pre-execution
views. Of 377 majority-labeled violations, 301 occur after their trajectory's
first positive step. This motivates an offline check of released monitor
predictions, **not** a paper greenlight or a claim of poor monitor performance.
No new GPU run is queued. Theory and video-removal audits likewise have not
yet qualified a paper-scale expansion.
The subsequent leave-one-expert-out check found first/later positive agreement
79.27%/83.53%, with the trajectory-bootstrap difference interval crossing zero.
Public model prediction files are explicitly uncommitted. The generic
first-boundary direction is parked, not promoted to a GPU sweep.
The [calibration follow-up](docs/MONITOR_CALIBRATION_NOVELTY_CHECK_20260904.md)
found direct prior work on session-level risk control and insufficient independent
safe episodes for a simple 5%-risk/95%-confidence certification claim. It also
clarifies that inconclusive evidence is not a hypothesis refutation.

### Current GPU window — September 4, 15:45 UTC

No GPU process remains active. Completed experiment evidence and trained adapters
are secured locally; no expansion is queued. The latest
[SDPO calibration decision](docs/SDPO_SINGLE_PROFILE_CALIBRATION_DECISION_20260904.md)
withholds training: this assay does not establish an informative improvement
stratum and has factual-preservation problems. This is not a paper-hypothesis
refutation. Further CPU-only candidate investigation can continue without this
instance. The public video-removal release audit found 180 complete triples,
but has not yet established valid effect-region ground truth or a paper result.

The paragraphs below retain the sequence of earlier checks in this window.

Subsequent [motion-interface diagnostic](docs/EP_INTERFACE_DIAGNOSTIC_RESULT_20260904.md)
completed and is secured locally: native video2/6, ordered images2/6, first-frame
control2/6, numeric oracle4/6. This does not qualify a larger motion experiment;
the original real-video hypothesis remains untested. The closer SDPO calibration
was launched but stopped before its first generation on a missing Jinja version;
the isolated-environment repair passed native template checks and a fresh-root
retry completed16 cases in284.87seconds, with zero parameter updates. Evidence
is secured and receipt-verified locally. The
[configuration audit](docs/SDPO_GENERATION_CONFIG_FORENSIC_20260904.md) establishes
sampled rather than intended greedy execution; factual/style review is complete,
and training is not approved. The GPU is idle.

The [read-only checkpoint diagnostic](docs/SDPO_FROZEN_FORWARD_DIAGNOSTIC_20260904.md)
completed all 192 forwards in 21.64 seconds, without training. Its complete
archive is local and remote/local SHA-256 matched. The
[forensic audit](docs/SDPO_FROZEN_FORWARD_FORENSIC_FINDINGS_20260904.md) verified
byte identities and scalar/norm arithmetic, but leaves gradient directions
unresolved (29 raw cosine discrepancies and 197 null-convention differences).
No tolerance was relaxed and no directional claim is made. Importantly, a valid
lowercase response can have almost zero probability for the original uppercase
first token: token displacement alone is not semantic failure. The separate
archived generation test established the teacher's preservation defect.
The original logits and frozen verifier remain preserved. A closer
single-profile, released-default SDPO reproduction is being prepared, not yet
running; it would qualify our experimental learner, not establish a new paper.

SDPOv3 completed: exact format success16/64 →0/64, factual correctness61/64 →64/64.
All64 updates and checkpoints are verified and saved locally. This shared-user
pilot failed, not the broader SDPO method or causal hypothesis; see
[result and scope](docs/SDPO_V3_RESULT_20260904.md). No feedback-corruption arm
is queued. The GPU was checked idle after completion; read-only diagnosis is
ongoing and all completed experiments' unique evidence is secured. The preceding
20,480-output coupling clarification completed and is secured locally:
[verified results](docs/COUPLING_CLARIFICATION_RESULT_20260904.md).
The early small-pair gain did not replicate strongly; the method is ~10% worse
than token-clock on the small pair after cost, with uncertain modest gains on
the larger pair. Park this heuristic, without claiming its true effect is zero.
An invalid prepared-manifest self-entry is disclosed; all substantive inputs
and separately pinned manifest bytes verified, including fresh reconstruction.
Earlier “idle”, “no queue”, and “safe to terminate” statements below describe
their historical runs, **not the current instance**. Preserve this instance until
the active work and its evidence transfer finish; target evidence cutoff is
15:40 UTC ahead of the user's approximately 16:21 UTC termination window.

Full-response SDPO's format positive control stopped **before training**:
explicit-format success 16/32, content correctness 32/32; hindsight-format
success 25/32, content correctness 32/32. All 24 initially mismatched responses
were corrected, but strict formatting requirements failed elsewhere. Its result
is `UNQUALIFIED_GENERATIVE_APPARATUS`, not a rejection of endogenous-feedback
learning. Complete evidence and the earlier pre-generation source-hash failure
are saved in `retrieved/sdpo_format_20260904T0902Z`; archive SHA-256
`a54c375b7e2546219a289aac0c9430f0cbd5c3fdd8b966a34a567635f8545446`.
The read-only verifier checked 21 manifest files and native tokenizer records;
it did not replay neural forwards or locally verify large model weights.
Any apparatus repair must be versioned prospectively with fresh cases; these
outputs and this failed qualification remain unchanged.

### Earlier developmental records

**New local candidate:** [native-marginal cross-tokenizer coupling](docs/BYTE_CLOCK_COUPLING_DEV_20260904.md)
investigates cheaper stochastic model comparisons, not improved model accuracy.
A finite-model DEV is replay-verified: byte-clock/grouped coupling fixes constructed
tokenization mismatches, but a counterexample doubles variance. Thus no general
improvement claim or paper green light. Native-decoder qualification completed:
16 outputs, 124 forwards, 162 seconds; hashes, decoding and saved-logit sampling
replay verified. Exact-option formatting was poor, not evidence of zero comprehension.
The [stochastic SQuAD DEV](docs/CPU_COUPLING_RESULT_20260904.md) completed on the
laptop: eight questions,16seeds,four policies,two small models (1,024completions).
Its descriptive F1 variance reduction is positive (~34% versus independent),
but the larger-model GPU result below is negative. Overall evidence is mixed,
not a paper green light; the bounded follow-up is described above.

**Subsequent GPU result:** the larger Qwen3-4B / SmolLM2-1.7B comparison completed
1,024 outputs in 84.33 seconds, with evidence retrieved and verified. Hierarchical
F1 difference variance was 1.058x independent and 1.252x the simpler token-clock
baseline; including measured cost, 1.084x and 1.328x. No practical win demonstrated;
park this heuristic. See [result audit](docs/GPU_COUPLING_RESULT_AND_BASELINE_AUDIT_20260904.md).
The CPU run above remains a separate developmental record, not a reason to
ignore this negative result. Both are retained for interpretation.

**Corrected UNDO training now complete:** [verified comparison](docs/UNDO_TRAINING_RESULT_20260904.md).
On32 longest histories: no adaptation23, terminalSFT24, canonicaldistillation28,
localrewrite24. Three matched256-update arms,410.82seconds; all checkpoints saved
locally and58manifest files verified. Park the local-rewrite method: it does not
beat canonical distillation. This is not evidence that all training failed.

**Latest completed comparison:** [OPDLM on-policy verification](docs/OPDLM_ONPOLICY_20260904.md)
finished 24 arithmetic prompts under four closed-loop policies plus 87 paired
verifier probes: 1,661 CPU forwards in 19.33 minutes. Evidence hashes and metrics
verify. Baseline and fresh checking score 24/24 under the predeclared permissive
last-number metric; both cache variants score 23/24. Strict answer-only compliance
is 0/24 in every arm. There is no demonstrated improvement over ordinary decoding.
Only 18/87 probed seeds belong to final answer numerals; baseline accuracy is at
ceiling, limiting correction conclusions. With direct prior-work overlap too,
this route is parked, not expanded. No OPDLM or GPU experiment queue remains.
All artifacts are preserved; the GH200 was live-checked idle after completion
of its previously secured runs and is safe to terminate for these experiments.

**Latest modern-decoder qualification:** [OPDLM cache DEV](docs/OPDLM_CACHE_DEV_20260904.md)
ran locally on pinned OPDLM-0.6B. Native and instrumented attention agree exactly,
but a 16-case Wikipedia continuation DEV does not show a useful late-cache
advantage (fresh 3/64 exact token matches; late-only also 3/64). Four native chat
prompts yield the expected facts with formatting/duplication problems. This
qualifies a local implementation for an on-policy diagnostic, not a paper.
160 paired forwards including an exact replay, plus 36 chat forwards; 25 combined
tests pass. All evidence preserved. No active GPU or CPU queue.

**Latest corrected-format result:** [short-right-context cached verification](docs/CACHE_SHORT_CONTEXT_20260904.md)
completed 1,408 CPU forwards on 128 new sentences. Fresh verification scores
60/128 versus 52/128 for corrected stale cache; 119/128 initial drafts are
lexical. The formal screen is inconclusive (19 fresh-recoverable errors versus
20 required), not a paper go. The next useful step is modern-decoder validation
and a drafting/verification tradeoff, not a larger BERT run to cross that cutoff.
Evidence verified; 23 combined tests pass. No active GPU queue.

**Latest natural-draft follow-up:** [public-text cached verification](docs/CACHE_NATURAL_DRAFTS_20260904.md)
completed 1,408 CPU forwards on 128 WikiText validation cases. Fresh masking
scores 73/128 versus 62/128 with corrected stale cache, but every initial draft
is punctuation: the truncated-context format fails the intended lexical-draft
premise. This is not a paper green light despite the numerical rule passing.
A separate 48-forward DEV check with three right-context tokens produces
23/24 lexical guesses and 13/24 correct; the repaired comparison is completed
above. All earlier evidence is preserved.

**Latest pretrained follow-up:** the [cached-verification BERT audit](docs/CACHE_VERIFICATION_BERT_20260904.md)
completed 288 CPU forwards on 24 frozen clozes. The official COVER source is now
located and its relevant information flow inspected. Candidate dependence
survives a pretrained network, but corrected-cache accuracy is 16/24 versus
14/24 for fresh masking; no useful clean-verification improvement is established.
Both candidate-dependent answer changes occur outside the clean-competent
subset. Seventeen combined tests pass; evidence verified, no active GPU queue.

**Latest candidate check:** a [cached-verification operator audit](docs/CACHE_VERIFICATION_AUDIT_20260904.md)
finds that exact per-layer diagonal correction does not ensure end-to-end
independence from a hidden candidate: information can return through other
positions at later layers. An analytic witness and random-network depth controls
are replay-verified; 13 tests pass. This is not an official decoder reproduction
or a pretrained-model performance result. A local trained-model diagnostic is
the next step, not an active GPU queue. The [diffusion scout](docs/DIFFUSION_SCOUT_20260904.md)
records close prior work and the limited novelty claim. No paper green light.

**Latest workflow audit:** [protocol events versus agent decisions](docs/AGENTUQ_DECISION_EVENTS_20260904.md)
compares held-out categorical baselines on 19,929 events / 6,500 agent decisions.
Protocol transitions explain much predictability, but agent-history information
still improves tool-choice prediction. Neither a generic workflow predictor nor
a blanket artifact critique is a new-paper lead. Results replay-verified,
26 relevant tests pass, no GPU work resumed.

**Latest prefix test:** [observable execution telemetry](docs/AGENTUQ_PREFIX_TELEMETRY_20260904.md)
has been evaluated at fixed 2/4/8-action clocks. It improves Brier over both
simple baselines in 5/18 settings (GPT telecom eight-action AUROC .573 -> .700),
but not consistently. Close structural-monitor and intervention-value papers
already exist; no generic monitor paper expansion is justified. Twenty-one
relevant tests pass; artifacts replay-verified. No GPU resumed.

**Latest matched-task follow-up:** a [cross-model selection audit](docs/AGENTUQ_MATCHED_SELECTION_20260904.md)
has run on the 556 public traces, with verified task/protocol matching and
task-held-out logistic calibration. Ordinary uncertainty features do not improve
selection over choosing the model with the better training-task success rate.
A terminal action-count baseline adds 7.02 points in retail only; this is
retrospective and not a new method or early intervention result. Seventeen
relevant tests pass. No GPU job or automatic expansion launched.

**Latest offline research:** the [agent-monitoring audit](docs/AGENT_MONITORING_RESEARCH_AUDIT_20260904.md)
checks an exact clustered-certification failure and inventories three public
datasets. All 556 AgentUQ v1.1 trajectories are downloaded, hashed and analyzed
locally, with a replay-verified descriptive baseline. No new model calls or GPU
jobs were launched. The statistical issue is known and no novel useful monitor
has been demonstrated; this is research progress, not a paper green light.

**Updated 4 September 2026. No candidate has earned paper expansion.** Phantom
Rollback and Reward Extinction Debt both completed, but their prerequisites
failed: these are **invalid assays/model organisms**, not universal negative
results. The independent queue resumed on the GH200 on September 4: LC0 smoke
completed with invalid text controls, EP0 smoke completed with weak oracle
controls (25% accuracy), and native-video hindsight completed with a valid
negative (zero observed past-state corruption in 48 pairs). All three runs
are secured locally. A 24-call diagnostic found exact token-ID/embedding
equivalence and unfinished reasoning under the old budget. The fresh 96-call
reasoning-enabled channel DEV is now verified: text controls 16/16, latent and
swapped-latent target accuracy 15/16, but the frozen global parse criterion failed
(81.25%; 16 no-message and two latent responses were token-limited). This is
encouraging communication evidence, not a valid gate pass or update result.
A separate released-C2C baseline completed and is secured: 128 public validation
questions, four arms, no training. Its strict-format report failed; a separate
audit using the published parser finds 56.25% fused accuracy versus 39.84% for
the receiver and 88.28% for the stronger sender. Conservative explicit-prefix
scoring still gives a 10.94pp fused improvement. This demonstrates a useful
existing interface, not our proposed update effect. No GPU job remains active;
text-transfer comparison and update training remain unrun. See the
[live queue and verified LC0 result](docs/INDEPENDENT_QUEUE_20260904.md).

The [Efference-Pair EP0](docs/EFFERENCE_PAIR_EP0_RUNBOOK.md) implementation is a
small synthetic camera-versus-object-motion apparatus pilot. Its 24-call smoke
is complete and did not justify the separate 450-forward pilot. It is **not**
the previously proposed ACaM/MotionBench G0. The real-video gate remains unrun.
The [latent-update baseline plan](docs/LATENT_UPDATE_BASELINE_PLAN.md) records
the current released-C2C baseline, source/model pins, and required controls.

The [data inventory](docs/DATA_AND_EXPERIMENT_QUEUE_20260902.md) and
[current executable queue](configs/research_queue_20260903.json) include five
initial stages: motion EP0, independent native-video hindsight, latent-channel
validity, hindsight/anchor feedback, and dialogue-retraction residue. Twelve MotionBench DEV
MP4s are local: 11 decode fully, including one cartoon; the remaining clip is
quarantined. These are development/media-audit examples, not a held-out result.
With `PYTHONPATH` set to `src`, run `python scripts/check_research_queue.py` to
verify prepared bytes without connecting to a GPU. No model-update robustness
experiment has been run. A sender-training/qualification stage is implemented
and CPU-tested, with disjoint data prepared. The paired old/new interface test
is also implemented, along with the conditional baseline repair comparisons. See the
[natural-update protocol](docs/C2C_NATURAL_UPDATE_PILOT.md) and
[paired-test protocol](docs/C2C_PAIRED_UPDATE_PROTOCOL.md) and
[repair protocol](docs/C2C_CACHE_REPAIR_PROTOCOL.md). None of these update/repair
stages has run on the GPU; renewed access confirmation and hardware validation
are required before launch.
LC0 only tests the communication prerequisite.

**Subsequent PI review: the same-input C2C update/repair chain is parked, not
queued to execute.** DroidSpeak and PrefillShare substantially overlap its
broad compatibility claim. A new paired audit finds only four questions where
fusion succeeds and the stronger sender fails, versus 45–52 in the reverse
direction (depending on parser); sender alone also has lower recorded generation
time. See the [novelty and utility review](docs/C2C_NOVELTY_UTILITY_REVIEW_20260904.md).
This is not a failed update experiment: no sender update has run. All prepared
code, data, and packages remain preserved.

The [five-idea PI review](docs/USER_IDEA_REVIEW_20260903.md) conditionally prioritizes
**Hindsight Is Not Counterfactual** and **Dialogue Retractions as Algebra**.
An additional [Hindsight sampling-law audit](docs/HINDSIGHT_SAMPLING_LAW_AUDIT_20260904.md)
verifies that own-response scoring and full reverse-KL distillation can produce
opposite updates under action-dependent feedback, including execution of the
pinned released losses on exact logits. Saved LM scores show the distinction
under imposed feedback channels. This is a diagnostic lead, not actual LM
training, a welfare result, or an established novel contribution.
A subsequent [pretrained CPU parameter probe](docs/HINDSIGHT_PARAMETER_PROBE_V1.md)
has now run: 66 forwards, eight backwards and two separate one-step adapters.
The primary directions are nearly parallel (cosine 0.999156); both score 7/8
on a tiny heldout set, versus 6/8 initially. This does not show a useful
practical estimator split or warrant expensive expansion. It is not full SDPO
training; all raw evidence and updated adapters are preserved locally.
The [matched factorial follow-up](docs/HINDSIGHT_FACTORIAL_PROBE_V1.md) is also
complete and verified: 82 CPU forwards, 16 backwards, zero updates. Removing
the initial preference statement does not explain the near-parallel directions.
Plain feedback wording changes their angle considerably, but the result is
highly sensitive to which of four tasks is included. No pooled condition has
opposite directions; this is prompt-sensitive geometry, not demonstrated
learning harm or a paper green light. No GPU job or automatic expansion is active.
Exact CPU checks establish a compact immediate-log ambiguity but disprove
automatic polarization in the proposed symmetric Bayesian example. The new
language assays have completed their smoke and full runs on the new GH200;
see [full results and conditional follow-up queue](docs/INTERACTION_FOLLOWUP_20260904.md).
The simple hindsight-anchor correction missed its criterion. UNDO showed an
initial 31–33-point long-history gap, but a fresh-history audit reduced this to
9.375 points, below its frozen 10-point robustness criterion. This is not zero
effect, but training expansion is held. **Neither full SDPO
training nor algebraic local-relation training is implemented or run**. Transport
certificates and learnability theory are parked; the proposed steering extension
has substantial prior-work overlap. These new apparatus stages are recorded
separately from the historical empirical-experiment count below.

**Latest completed local work:** the [capability-controlled feedback-learning pilot](docs/HINDSIGHT_COMPETENCE_PILOT_V1.md)
completed and was verified: 1,250 forwards, 544 backwards, 16 supervised warmup
and 64 feedback-stage updates. Qualification improves from 16/32 to 30/32,
but the checkpoint starts at only 36/64 on differently rendered heldout cases.
Under copying feedback, own-response and anchor-only finish at 36/64, full KL
and projected full KL at 40/64. Projection does not improve over full KL, and
no robust harmful-feedback/corrective-learning result is established. This is a
restricted categorical, frozen-teacher synthetic pilot, not full SDPO. All local
runs have finished; there is no active GPU or CPU experiment queue.
An independent [flow-uncertainty control](docs/FLOW_UNCERTAINTY_NULL_CONTROLS.md)
has an exact checked example where outputs remain unchanged while a velocity
disagreement score rises. A [finite-data affine-flow pilot](docs/FLOW_ENSEMBLE_PILOT_V1.md)
has now run and been verified: endpoint-distance acquisition loses to the
existing velocity score in all eight seed pools. This does not support the
proposed practical improvement. Its varying-label-count selection task is also
too easy to discriminate mechanisms well. Neural/VLA validation remains unrun;
no remote GPU expansion is queued.
A subsequent [equal-label-count flow control](docs/FLOW_FIXED_COUNT_V1.md)
also missed its practical-improvement criterion (whitened endpoint energy wins
4/8 pools, .820 times the original velocity score's mean acquisition gain).
The proposed uncertainty-replacement paper route is parked, not expanded.

See the [researcher journal](docs/RESEARCH_JOURNAL.md) for hypothesis, design,
numbers, evidence paths, failed launches, interpretation limits, and next steps;
the [new-idea review](docs/IDEA_REVIEW_20260902.md) records the Mostik/latent-channel
collision search and the next hypotheses. Historical protocols below are retained
for reproducibility, **not instructions to restart retired experiments**.

## Experiment ledger

Dates identify runs/retrievals, not necessarily publication. `pp` means percentage
points. A failed validity prerequisite is different from a valid negative effect.
This ledger has 15 empirical stages across 13 directions (Stage 1/DID share one
direction; RAG developmental/G0b share another). CPU fixtures, retries, and
literature-only candidates are not additional scientific experiments.

| Date | Experiment and hypothesis | Observed result | Research decision |
| --- | --- | --- | --- |
| Aug 19 | Under Extinction Stage 1: distinguish genuinely rewarded from proxy-rewarded policies using passive revaluation | Acquisition learned; DEV comprehension, selective response, reversal and robustness checks failed | Stop original formulation; locked TEST remains closed |
| Aug 25 | DID-v1.1.1: diagnose retention, update integration, planning and A/B interface | 76,800 score rows; mean label-equivariance error 0.1897; semantic label-swap agreement 75.09% | Label-interface contamination; no controller conclusion |
| Aug 25 | Provenance authority: self-authored material gains excess authority | Self-minus-external effect **0.0277 pp**, vs required 10 pp | Stop this formulation |
| Aug 25 | Response-interface invariance: equivalent action encodings change decisions | **0%** selection disagreement; probability spread **0.1051 pp** | Stop this formulation |
| Aug 25 | Hybrid memory: recurrent state materially carries a retained constraint | Recurrent intervention **0.168 logits** (bar 0.50); attention K/V control **14.449** | Small real effect, insufficient for proposed mechanism |
| Aug 28 | Recency G0: learned recency direction causally controls switching | All registered checks failed both seeds; corrected cue-matched effects **−0.072/+0.704 pp** | Kill candidate; no G1 |
| Aug 28 | Recipe-invariant J0: direction agreement predicts held-out causal transfer | Strong source agreement, held-out steering **−0.095/−0.224 pp** | Kill candidate; no external replication |
| Aug 28–29 | Semantic ancestry developmental run: source rewriting creates apparent diversity | Qwen snapshot **6,720 rows**; role/style confounds; Mistral stopped at user request | Developmental only; neither pass nor kill |
| Aug 29 | Role-separated semantic ancestry G0b: ancestry-specific failure plus history-aware remedy | Collapse contrast **+21.7 to +53.3 pp**; one specificity failure; remedy missed its MMR criterion in all four cells | Kill exact candidate, retain mechanism clue |
| Aug 29–30 | Effect-consistency uncertainty: executed successor-state agreement improves uncertainty/routing | **3/10** required checks passed across Qwen and gpt-oss | Kill candidate |
| Aug 30 | Visual patch phase: grid alignment causes specific errors remedied by phase ensemble | No thin-feature specificity/phase lock; ensemble **−7.5 pp Qwen, −17.5 pp Gemma** | Kill candidate |
| Aug 30 | Critique Oracle: actionable guardrail feedback induces recovery/gaming tradeoff | **0 fabricated authorization evidence**; safe-recovery gains **+35.4/+45.8 pp** vs opaque feedback | Recovery benefit, but hypothesized gaming tradeoff absent; kill candidate |
| Aug 30 | Reward-hack early warning (CPU, public trajectory): variance warns earlier than ordinary hack rate | **10,240 rollouts/40 checkpoints**; both AUROCs 1.0; advantage **0**; false alarms **26.7%** | Developmental fail; no new RL |
| Sep 1 | Phantom Rollback: ambiguous restore availability induces premature irreversible actions | Availability effect **0**; Qwen comprehension **12/16**, Gemma strict parse **0/16** | `INVALID_ASSAY_COMPREHENSION_OR_CAPABILITY`; no expansion |
| Sep 1 | Reward Extinction Debt: aligned former shortcut learner reacquires faster than clean control | Induction **+97.69 pp**; matching failed; debt AUC difference **−0.0174**, CI **[−0.0967, 0.0621]** | `INVALID_MODEL_ORGANISM_FORMATION`; no expansion |

Read [the journal](docs/RESEARCH_JOURNAL.md) before interpreting a short table
entry. Raw outputs/checkpoints remain under ignored `retrieved/` and `artifacts/`;
public summaries are not a substitute for those archives. Never commit private
answer keys, credentials, or restricted media.

## Historical Under Extinction implementation

The CPU oracle pipeline passes its synthetic consistency checks. The first paid
Qwen3.5-9B Stage-1 run completed on 19 August 2026 and **failed its registered DEV
gate**. Both reward-acquisition arms learned their respective objectives on late
TRAIN conflicts, but the passive-revaluation assay failed comprehension,
channel-specificity, reversal, and within-cell robustness gates. Its immutable
report and complete checkpoint archive are preserved under the Git tag and GitHub
release `stage1-dev-20260819-failed`. The locked TEST split has not been opened,
Gate C has not been attempted, and there is no claim that this work guarantees
aligned or symbiotic AGI.

The separately frozen
[`DID-v1 post-failure diagnostic`](docs/DEV_DIAG_PREREGISTRATION.md) tested
whether the failure arose from held-out objective retention, static causal
parsing, passive-update integration, objective–planner composition, or the A/B
interface. DID-v1 uses new DEV-only cases and frozen checkpoints. It cannot revise
the failed registered decision or authorize TEST.
Its separate inference contract audits all 19,200 prompts with the pinned
tokenizer before loading a model, freezes a 768-token ceiling above the exact
745-token corpus maximum, and fails instead of truncating; the historical bridge
configuration remains unchanged.

The paper's first paid experiment was the **same-environment RL bridge**. Starting
from paired copies of one base model, a G-RL policy learned from genuine outcome
reward and a P-RL policy learned from proxy/evaluator reward in the same
environment, with matched prompts, opportunities, update counts, and random
seeds. The registered extinction assay did not distinguish those policies
selectively on development data, so replication and locked-test evaluation remain
unauthorized.

The unchanged base model is the required negative control. A random-reward policy
may be run only as a budget-permitting exploratory control; it is not a missing
preregistered arm when omitted. Locked E2 replicates the two pure G/P arms only.
Mixture and open-set model tests begin in E3 and cannot be claimed from E2.

Bridge DEV/TEST maps are held-out worlds with disjoint nonce outcomes. They share the acquisition environment's causal schema but are not literal previously visited trials; the project states this explicitly instead of treating E1 as classical same-outcome devaluation.

The older symbolic SFT organisms remain useful as CPU/oracle apparatus calibration and code-path tests. Their labels directly encode a controller, so training a large symbolic matrix would provide weak scientific evidence and is not the default paid plan.

The project only becomes a serious paper if the assay later predicts unseen reward hacking beyond all of these baselines:

- ordinary behavior and confidence;
- current overt hack rate;
- direct genuine-versus-proxy conflict choices;
- genuine/proxy comprehension;
- PRIME-style proxy-gap knowledge;
- simple prompt counterfactuals;
- comparative instrumental-intervention profiles.

If the fingerprint adds no held-out predictive information, the project stops.

An independent [`response-interface invariance gate`](docs/INTERFACE_INVARIANCE_FEASIBILITY_PROTOCOL.md)
was prepared on branch `codex/interface-invariance-g0` and completed on August 25
with `STOP_INTERFACE_INVARIANCE_LINE`. It tests a different
measurement-validity hypothesis with a new corpus and reads none of the bridge
or provenance results. It is a bounded feasibility study, not a continuation of
either retired paper claim; its measured result is recorded in the ledger above.

## What is implemented

The primary bridge includes:

- paired G-RL and P-RL policies that act in one two-stage environment but are
  optimized on different realized reward channels;
- six split-disjoint renderer formats and disjoint train/DEV/locked-TEST nonce
  lexicons;
- a semantic-versus-neutral channel-identity ablation, fully crossed with
  renderer, channel order, action label, intervention family, and control mode;
- passive value revaluation, passive transition revaluation, unreachable shams,
  active no-switch controls, and one unrewarded first choice;
- six fixed formal training checkpoints per arm, exact A/B likelihoods,
  legal-choice mass, and a
  one-token unconstrained parse diagnostic, with no LLM judge;
- resume-safe cumulative acquisition curves plus a separately preregistered
  trailing-update acquisition window (50 formal updates; one smoke update), with
  continuation gates using only the latter across cue×conflict cells;
- paired seed×world analysis, crossed bootstrap uncertainty, and continuation
  gates evaluated within seed×cue and cue×intervention-family cells (including
  role/renderer retention and unchanged-base neutral-cue selectivity), plus a
  trajectory-held-out prospective-analysis module;
- hash-bound data, configs, optimizer specifications, adapters, checkpoints,
  predictions, runtime manifests, resumable training, exact-hardware GH200
  preflight, budget
  wrappers, artifact collection, and an independent termination watchdog.

The package also retains the older symbolic intended/proxy/cue-controller fixtures
as optional apparatus calibration. They are not part of the default paid workflow
and cannot establish the paper's learning claim.

## Local validation

From PowerShell:

```powershell
cd <path-to-Align_Paper>
python -m pip install --requirement requirements/cpu-test.lock
python -m pip install --no-deps --editable .
./scripts/cpu_smoke.ps1
```

Or without installing the package:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m pytest tests -q
python -m under_extinction --config configs/bridge_smoke.yaml bridge-build
python -m under_extinction --config configs/bridge_smoke.yaml bridge-oracle --split dev
python -m under_extinction --config configs/bridge_pilot.yaml bridge-build
python -m under_extinction --config configs/bridge_pilot.yaml dry-run
```

The symbolic smoke oracle and bridge oracle are deterministic apparatus checks. Any wording that treats either as an empirical result is a provenance error.

The paid runtime uses released packages only: Transformers 5.15.0, PEFT 0.20.0,
Accelerate 1.14.0, Hugging Face Hub 1.27.0, tokenizers 0.22.2, and safetensors
0.8.0. It never installs a Git branch or silently compiles optional DeltaNet
extensions on the paid machine. The first run fixes `use_kernels=False` and
requires the released Transformers PyTorch fallback so smoke and Stage 1 use the
same deterministic backend contract.

## Paid run sequence

1. Read the bridge-first [PAPER_PLAN.md](docs/PAPER_PLAN.md), [PREREGISTRATION.md](docs/PREREGISTRATION.md), and [THREATS_AND_KILL_CRITERIA.md](docs/THREATS_AND_KILL_CRITERIA.md). [PILOT_PROTOCOL.md](docs/PILOT_PROTOCOL.md) is retained only as the explicitly archived symbolic-calibration design, not as the paid protocol.
2. Build and verify `bridge_smoke.yaml` and `bridge_pilot.yaml` locally.
3. Make a slim, checksummed bridge bundle and follow [LAMBDA_RUNBOOK.md](docs/LAMBDA_RUNBOOK.md).
4. Arm an independent termination watchdog and export its absolute deadline. Every bridge script reserves at least 30 minutes for retrieval.
5. Run `preflight_bridge.sh`; it exercises the exact pinned Qwen3.5-9B text model, in non-thinking mode, with a one-update bridge. It hash-verifies the transferred formal corpus and its frozen workload profile, but does not parse or expose formal development or locked-test records.
6. Run `run_bridge_stage1.sh`; it trains one paired G-RL/P-RL seed and evaluates **DEV only**.
7. Stop and inspect the report, throughput, and cost. Replication is a separate command and requires both a passing Stage 1 gate and an exact human-created approval file.
8. Only then run `run_bridge_replication.sh`, which trains the remaining paired seeds and opens locked **TEST** once.
9. Retrieve and hash artifacts before terminating the instance.

Do not spend paid GPU time on the twelve-adapter symbolic SFT matrix unless a later, explicitly justified ablation requires it. Its CPU oracle outputs validate mechanics, not the central learning claim.

The frozen paid-hardware contract is one Lambda `gpu_1x_gh200` instance:
`aarch64`, device name `NVIDIA GH200 480GB`, one compute-capability-9.x GPU,
and at least 90 GiB of CUDA-visible memory (the offering has nominally 96 GB
HBM). At the displayed rate recorded on 2026-08-19, $2.29/hour, reserving 15%
gives a 37.12-hour cap from a $100 nominal budget or 74.24 hours from $200,
before tax. These are hard ceilings, not runtime estimates. Confirm the displayed
rate at launch; the config is not a billing authority. Qwen3.5-9B is materially
more expensive than the retired 1.5B prototype, so only measured exact-model
preflight throughput can authorize Stage 1.

## Layout

```text
configs/       frozen smoke and formal-pilot specifications
docs/          protocol, preregistration, threats, and Lambda runbook
requirements/  CPU-test and paid CUDA-runtime dependency locks
scripts/       local smoke, paid stages, telemetry, retrieval, watchdog
src/           generator, trainer, scorer, analysis, manifests, CLI
tests/         offline unit and end-to-end oracle tests
artifacts/     generated outputs (ignored by Git)
deployment/    slim transfer/results archives (ignored by Git)
```

## Interpretation boundary

“Controller” here means a learned action-control signature under specified interventions. It does not mean a metaphysical terminal goal, subjective desire, or conscious motivation. “Extinction-style” means the test choice supplies no reward, correction, observed consequence, or subsequent update; it does not claim that frozen transformer inference is biologically identical to animal extinction learning.

## Reward-Seeking Extinction Debt candidate

The repository also contains a separately frozen, bounded model-organism gate
for testing whether ordinary alignment leaves unusually rapid reacquisition of
a proxy-rewarded shortcut. Its protocol, decision contract, and interpretation
boundary are in
[`docs/CANDIDATE_REWARD_EXTINCTION_DEBT.md`](docs/CANDIDATE_REWARD_EXTINCTION_DEBT.md),
with the exact GH200 and offline-verification sequence in
[`docs/REWARD_EXTINCTION_DEBT_G0_RUNBOOK.md`](docs/REWARD_EXTINCTION_DEBT_G0_RUNBOOK.md).
It does not share data, checkpoints, or a decision with Phantom Rollback.
The completed September 1 run returned `INVALID_MODEL_ORGANISM_FORMATION`;
see the journal rather than treating the historical runbook as a pending launch.
