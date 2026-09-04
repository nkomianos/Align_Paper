# AI Research Journal — ICLR 2027 exploration

This repository preserves the hypotheses, implementations, failed attempts,
measurements, and decisions from our ICLR 2027 research exploration. Its original
project was **Under Extinction**:

> Can an extinction-style intervention fingerprint identify what controls a language agent's actions when ordinary behavior is matched—and predict later specification gaming better than current behavior or knowledge probes?

It now also contains independent alignment, agent-security, and vision candidates.
No historical OSH adapter or old OSH paper artifact is imported or packaged.

## Scientific status

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

**Current local work:** a [capability-controlled feedback-learning pilot](docs/HINDSIGHT_COMPETENCE_PILOT_V1.md)
is running on the laptop CPU (not the idle remote GPU). Sixteen supervised
warmup updates improved separate qualification accuracy from 16/32 to 30/32,
passing its fixed capability rule. Eight short feedback/anchor comparison arms
follow from that same checkpoint; final results are pending. This is a restricted
categorical, frozen-teacher synthetic pilot, not a full SDPO reproduction.
An independent [flow-uncertainty control](docs/FLOW_UNCERTAINTY_NULL_CONTROLS.md)
has an exact checked example where outputs remain unchanged while a velocity
disagreement score rises. A [finite-data affine-flow pilot](docs/FLOW_ENSEMBLE_PILOT_V1.md)
has now run and been verified: endpoint-distance acquisition loses to the
existing velocity score in all eight seed pools. This does not support the
proposed practical improvement. Its varying-label-count selection task is also
too easy to discriminate mechanisms well. Neural/VLA validation remains unrun;
no remote GPU expansion is queued.

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
