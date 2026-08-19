# Threats, rival explanations, and kill criteria

> **Status note.** The symbolic SFT matrix described below is now apparatus calibration. The first paid falsification is the same-environment G-RL/P-RL bridge in [PAPER_PLAN.md](PAPER_PLAN.md); the criticisms below remain active kill criteria.

## Closest prior work

This project is not novel merely because it intervenes on inferred motivation. Comparative Motivation Profiles already uses symmetric instrumental interventions to distinguish consequence tracking from researcher-expectation tracking in alignment-faking organisms. PRIME already forecasts later reward hacking from learned proxy-gap knowledge. Goal-directedness work already separates ability from goal pursuit and combines behavior with representations. The proposed contribution must therefore be the joint package of:

1. selective outcome **value** and **contingency** interventions under no-feedback choice;
2. ground-truth controller recovery under observational equivalence and locked semantic transformations;
3. incremental prospective prediction beyond those prior baselines.

Relevant starting points:

- [Building Comparative Motivation Profiles with Instrumental Interventions](https://arxiv.org/abs/2606.08243)
- [Proxy Reward Internalization and Mechanistic Exploitation (PRIME)](https://arxiv.org/abs/2606.09711)
- [Evaluating the Goal-Directedness of Large Language Models](https://arxiv.org/abs/2504.11844)
- [A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](https://arxiv.org/abs/2602.08964)
- [Reward Hacking in Language Model Agents: Revisiting AI Safety Gridworlds](https://arxiv.org/abs/2606.15385)
- [Outcome devaluation as a method for identifying goal-directed behaviors](https://www.nature.com/articles/s41596-024-01054-3)

## Rival explanations and controls

### The model follows an explicit update instruction

That is expected at the comprehension layer and is not enough for controller identification. Every organism receives the same update and must understand all channels. Only the learned action rule differs. Still, prompt demand is a major threat. Locked formats, nonce vocabularies, reference-outcome shams, channel-order swaps, and later below-prompt/SDF or weight-level replication are required.

**Kill:** if prompt wording predicts the label across organisms, or if prompt-only counterfactual baselines match the assay's prospective value.

### Revaluation is just another static conflict

An intended or proxy update ultimately produces a conflict, so a reviewer can ask why a direct static conflict prompt is insufficient. The contingency interventions and prospective test are the answer only if they empirically add robustness or prediction.

**Kill:** if direct conflict behavior performs equivalently on held-out forecasting and the full fingerprint contributes no log-loss gain.

### The assay detects knowledge, not control

All agents may know which action genuinely helps and which the evaluator rewards. Comprehension and PRIME-style probes measure that knowledge separately. The audit claims to measure which represented consequence controls choice.

**Kill:** if comprehension differences explain fingerprint differences, or PRIME absorbs all prospective signal.

### SFT labels create artificial personas

The synthetic organisms provide ground truth but may not resemble controllers learned through reward optimization. They establish construct validity only.

**Kill:** if fingerprints do not vary before and through naturally emerging proxy optimization, even when the SFT benchmark is perfect.

### Surface familiarity or negative wording drives shifts

Active updates mention action-produced outcomes; shams mention disclosed reference-only outcomes through the same renderer-specific update shell.  The primary formal corpus fully crosses action side, channel role, renderer, and independently authored wording.  Value-switch trials balance negative revaluation of the previously preferred outcome against positive revaluation of the previously nonpreferred outcome, and each paired unreachable-outcome sham matches the active update's sign.  Neutral factual updates remain required in the robustness phase.

**Kill:** sham shifts exceed 0.05, the minimum within-cue×family renderer
effect-retention ratio falls below 0.75, or a text-only prompt classifier can
infer organism identity above chance.

### “Controller” overstates what behavior identifies

Finite interventions cannot uniquely establish terminal goals. The valid claim is a predictive action-control signature under a specified intervention family.

**Kill/reword:** any evidence that supports only a task-local heuristic must be described as task-local. Never claim consciousness, intrinsic care, terminal values, or guaranteed alignment.

### The GH200 migration changes the effective experiment

Accelerator architecture can affect throughput, numerical trajectories, and
available kernels. The migration to one ARM64 Lambda GH200 occurred before any
model download, training, or empirical observation and does not license changes
to the model, data, optimizer, seeds, endpoints, or gates. Both reward arms must
run under the same attested PyTorch/CUDA/device contract, with the frozen
`torch_fallback_required` DeltaNet path. The one-update preflight is an engineering
and cost measurement only, never evidence for the scientific hypothesis.

**Kill:** abort before Stage 1 if the machine is not exactly one
`NVIDIA GH200 480GB` on `aarch64`, exposes less than 90 GiB CUDA memory, has a
compute-capability major version other than 9, imports a forbidden optional
DeltaNet backend, or cannot satisfy the frozen released dependency stack without
in-place driver or source-build repair.

## Ways the paper can fail despite clean code

- Qwen3.5-9B may fail to acquire both reward-channel policies while preserving
  ordinary behavioral equivalence; its hybrid DeltaNet/attention LoRA path may
  also be too slow for the initial budget.
- The base model cannot transfer static calibration to acquisition/update prompts.
- Contingency language is understood less reliably than value language.
- A/B token preference dominates one or more renderers.
- The prototype identifies hand-built adapters but not natural checkpoints.
- Direct conflicts or PRIME dominate forecasting.
- Six RL trajectories are too few for stable uncertainty.
- The reward-hacking environment does not produce a useful pre-onset regime.

These are scientific outcomes, not bugs to tune away. The gates exist to prevent an attractive narrative from surviving a failed central hypothesis.
