# Efference-Pair EP0: apparatus-first motion pilot

Status: **CPU validated; ready for the first GPU smoke, not a completed VLM result**.
Date: 2026-09-02. Code version: `efference-pair-ep0-v3`.

## Scope and preregistration boundary

This is a **new, explicitly separate synthetic pilot** before the historical
[96-item real-video proposal](CANDIDATE_EFFERENCE_PAIR.md). EP0 uses 224-pixel
canvases and planar translations; the real-video proposal specified 448 pixels,
ACaM/MotionBench and additional controls. No threshold or dataset in that formal
protocol has been silently changed. Its real-data slice is not sealed or run.

The synthetic world has a textured stationary plane and a textured red square,
viewed by an orthographic camera. Each camera/object trajectory is one of left,
right, up, down or stationary; movement is exactly two pixels per frame. All
25 combinations appear. Each asks two questions: camera translation and square
motion relative to the stationary world. Common camera interventions share the
same texture/object/world-motion configuration. Camera and world-object motion
are independent factors; cancellation and static controls are present.

The RGB-only estimator uses frozen Farneback flow and robust partial-affine
RANSAC, then subtracts the global field. It does not receive simulator masks,
motion labels or answers. Simulator masks/vectors are available only to the
separate oracle-field condition, numeric diagnostics and answer-key generator.
Global arrows invert apparent background flow to depict camera translation;
residual fields and RGB are warped into the initial camera frame.

## Exact workload

Every case has 16 canvases at 224x224, with neutral time stamps. In particular,
extra-RGB/derived sequences explicitly disclose repeated or restarted times;
we do not let a misleading chronological layout manufacture a treatment gain.
Questions/options are identical across
conditions. Option order is deterministically permuted independently of treatment.
Greedy Qwen3-VL-8B-Instruct, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, at most eight new tokens.

| Condition | Input |
| --- | --- |
| native_rgb | 16 ordered RGB frames |
| rgb_layout | 8 common anchors + 8 additional RGB frames |
| raw_flow | Anchors + 8 undecomposed flow views |
| global_only | Anchors + 8 camera-translation arrow views |
| residual_only | Anchors + 8 stabilized residual overlays |
| joint | Anchors + 4 global/4 residual, interleaved at shared times |
| oracle_joint | Same interface, simulator ground-truth motion fields instead of estimated fields |
| sign_reverse | Same anchors, both estimated fields sign-reversed |
| sham | Same anchors, deterministic other-scene fields |

The **full pilot is 25 scenes × 2 questions × 9 conditions = 450 forwards**.
The **smoke is 24 forwards**: both questions on four frozen scenes, using native,
joint and oracle. It cannot return a scientific pass. Modes require separate
fresh output roots; no automatic smoke-to-full chain.

The processor's image grids and actual image-token count are audited for *every
planned input before the first scored forward*. Any discrepancy stops the run;
there is no retrospective token normalization. A shared prompt makes condition
labels/wording unable to encode the answer. Completion parsing accepts exactly
one stripped A–E letter; raw text is preserved.

Sham in EP0 is a deterministic wrong-scene control, **not** the magnitude-matched
sham specified for real videos. The oracle replaces fields, not the entire
stabilization pipeline. Constant-velocity temporal shuffling would preserve the
motion information, so EP0 makes no temporal-order causality claim and does not
use that uninformative test as a gate. These are deliberate scope limits.

## CPU validation performed

- Generated all 25 scenes, 450 cases, 3,600 PNG views and five metadata files.
- All scenes pass the unchanged numeric tolerances: scene-median global vector
  error <0.35 pixels, residual error <0.50 pixels.
- Maximum across scene-median errors: **0.01942 pixels global** and **0.04303
  pixels residual** (rounded upwards).
- Unit tests cover camera-sign/cancellation, conservation `raw = global + residual`,
  determinism, common anchors, exact output completeness, strict parsing,
  manifest tampering, and smoke/ceiling/negative-result decisions.
- **28 targeted CPU tests passed**: 17 Efference-Pair, 3 latent-fixture and 8
  existing visual-hindsight tests. These include mocked outcome tests, not LLM runs.
- The initial v1 fixture failed residual-accuracy tests; its evidence is retained.
  v2 fixes fractional-step rasterization and weak object texture; v3 adds truthful
  timestamps to remove layout ambiguity. All precede any VLM run; v1/v2 are retained.

Prepared v3 corpus (local, ignored):
`artifacts/efference_pair_ep0_20260902_v3/`.
Its `MANIFEST.json` SHA-256:
`b111de93c0da843e3e24723e5cd7dc2e63d1018576bc81ab8c7b5d140d3ab97e`.
**No model inference, model accuracy, causal VLM effect or paper success is
claimed from these CPU checks.**

## Decision policy

Smoke: report parsing, native/oracle/joint accuracy and median forward latency.
Inspect failures and do not extrapolate scientific conclusions from four scenes.
Proceed to the full pilot only if the runtime works, formatting is reliable,
oracle displays are usable, and the measured time is acceptable to the user.

Full pilot requires parse rate >=95% and oracle accuracy >=80%; otherwise
`STOP_EP0_INTERFACE_OR_CAPABILITY`. Native accuracy >=90% yields
`STOP_EP0_CEILING_NOT_INFORMATIVE`, not a claimed method failure.

To return `READY_TO_DESIGN_REAL_VIDEO_G0_NOT_PAPER_PASS`, all of the following
must hold: joint gain >=8 pp over native; >=4 pp over extra RGB and raw flow;
>=4 pp gain in both strata; global and residual selectivity >=4 pp each; joint
beats sham by >=8 pp; sign reversal increases reversed-answer choice by >=10 pp;
static accuracy loses at most 2 pp. Otherwise `STOP_EP0_NO_FULL_BENCHMARK_SPEND`.
These pilot bars are not the real-video gate's registered thresholds.

All 25 scenes remain in analysis. Confidence intervals are descriptive: there
are only **five independent object/background groups**, each observed under five
camera interventions. Bootstrap whole groups, not frames or 450 correlated
forwards. No significance or generalization claim is justified by this pilot.

## Local commands

Use an isolated apparatus environment; the historical package pins NumPy <2,
whereas this tested OpenCV build uses NumPy 2. **Do not upgrade old frozen run
environments or install the whole repository's GPU extras into this environment.**

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/test_efference_pair.py -q
# Fresh output only. The already prepared v3 root should not be regenerated.
python -m efference_pair.runner prepare --output artifacts/ep0-FRESH-ID
```

`requirements/efference_pair_cpu.txt` records the tested local CPU packages.
On a new GH200 use a separate environment with hardware-compatible CUDA PyTorch
and native Qwen3VL Transformers support; pin/record the resolved environment
before inference. CUDA loading is **not tested on this laptop**. No broad
dependency upgrade or replacement of another experiment environment is allowed.

## Remote sequence after the user supplies an instance

1. Transfer source plus the prepared corpus (not credentials/private unrelated
   data); compare local/remote corpus manifest digests and validate every file.
2. Verify architecture/CUDA, native Qwen3VL loading, disk capacity and exact model
   revision. Use the existing SSH key; this public model does not require a new
   paid API account. A gated second family is not part of EP0.
3. Set `PYTHONPATH` to this repository's `src`; do not install incompatible legacy
   package extras. Use `EP_MODE=smoke`, then run:

```bash
export EP_PYTHON=/path/to/isolated-ep0-env/bin/python
export EP_PREPARED=/home/ubuntu/frozen_inputs/ep0-v3
export EP_RUN_ROOT=/home/ubuntu/ep0-smoke-FRESH-ID
export EP_MODE=smoke
bash scripts/run_efference_pair_ep0_remote.sh
```

4. Check the 24-forward evidence and measured median. The extrapolation is
   `450 × median_seconds / 3600` hours, **excluding** download, load, transfer and
   processing audit time. We have no honest measured GH200 ETA before this step.
5. If approved, repeat with `EP_MODE=full` and a new root. Preserve smoke evidence.
6. Retrieve complete evidence, compare manifests, then analyze outside the root:

```bash
python -m efference_pair.verify --root /path/to/retrieved-root --output /path/to/NEW-report.json
```

The runner snapshots inputs, source, budget audit, runtime metadata and raw
completions, flushes every record, seals completed or failed runs, and refuses
overwrites. The verifier is read-only with respect to evidence. Retrieve failed
and partial runs too. Never silently retry into the same directory.

## What is still needed for a real paper gate

Licensed and reviewed ACaM/MotionBench clips, fixed real-data slice, true
motion-boundary and fixed-reference tracking baselines, non-translation camera
tests, full causal-control implementation, and independent model-family
confirmation. None is replaced by synthetic success. That work is conditional on
EP0 being informative; no multi-GPU instance is warranted now.
