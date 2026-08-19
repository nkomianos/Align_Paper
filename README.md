# Under Extinction

This package implements the first, deliberately falsifiable phase of a proposed alignment paper:

> Can an extinction-style intervention fingerprint identify what controls a language agent's actions when ordinary behavior is matched—and predict later specification gaming better than current behavior or knowledge probes?

This standalone repository contains only the current Under Extinction project. No
historical OSH adapter, hidden worktree, or old paper artifact is imported or
packaged.

## Scientific status

The CPU oracle pipeline passes its synthetic consistency checks. That means generation, interventions, pairing, scoring, bootstrap analysis, and gates agree with known ground truth. It is **not model evidence**. No RL organism has been trained in this checkout, the locked bridge test has not been opened, Gate C has not been attempted, and there is no claim that this work guarantees aligned or symbiotic AGI.

The paper's first paid experiment is now the **same-environment RL bridge**. Starting from paired copies of one base model, a G-RL policy learns from genuine outcome reward and a P-RL policy learns from proxy/evaluator reward in the same environment, with matched prompts, opportunities, update counts, and random seeds. The extinction assay must distinguish those independently learned policies on development data before any replication or locked test is authorized.

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
  predictions, runtime manifests, resumable training, H100 preflight, budget
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

Do not spend H100 time on the twelve-adapter symbolic SFT matrix unless a later, explicitly justified ablation requires it. Its CPU oracle outputs validate mechanics, not the central learning claim.

At the rate recorded on 2026-08-16, $3.29/hour for a one-GPU H100 PCIe instance, reserving 15% gives a 25.84-hour cap from a $100 nominal budget or 51.67 hours from $200, before tax. Confirm the displayed rate at launch; the config is not a billing authority. Qwen3.5-9B is materially more expensive than the retired 1.5B prototype, so only measured exact-model preflight throughput can authorize Stage 1.

## Layout

```text
configs/       frozen smoke and formal-pilot specifications
docs/          protocol, preregistration, threats, and Lambda runbook
requirements/  CPU-test and H100 dependency locks
scripts/       local smoke, paid stages, telemetry, retrieval, watchdog
src/           generator, trainer, scorer, analysis, manifests, CLI
tests/         offline unit and end-to-end oracle tests
artifacts/     generated outputs (ignored by Git)
deployment/    slim transfer/results archives (ignored by Git)
```

## Interpretation boundary

“Controller” here means a learned action-control signature under specified interventions. It does not mean a metaphysical terminal goal, subjective desire, or conscious motivation. “Extinction-style” means the test choice supplies no reward, correction, observed consequence, or subsequent update; it does not claim that frozen transformer inference is biologically identical to animal extinction learning.
