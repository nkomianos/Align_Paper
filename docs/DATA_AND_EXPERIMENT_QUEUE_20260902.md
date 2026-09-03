# Data in hand and the next GPU queue

Updated 2 September 2026. No new scientific model results. This is executable
preparation, not a claim that three publishable papers have been discovered.

## Data actually on this laptop

| Dataset | Prepared | What remains unestablished |
| --- | --- | --- |
| Efference-Pair EP0 | 25 controlled 16-frame scenes; nine presentations; 450 question/presentation records | VLM accuracy, general 3-D motion, real-video improvement |
| Visual hindsight | 48 paired worlds, byte-identical pasts and matched changed futures; 240 calls | Any VLM effect, natural-video prevalence, a publishable remedy |
| MotionBench DEV | 12 MP4s, 36,001,980 bytes, pinned upstream checksums | Complete label review and an untouched held-out evaluation |
| Latent channel LC0 | 64 nonce-relation pairs / 128 cases; separate sender/receiver facts | A useful learned-model channel or model-update robustness |

Of the 12 benchmark clips, **11 fully decode**. Their preview inspection shows
**10 live-action clips and one cartoon**. One other clip is quarantined:
`uhCc1MlfC9hHyJe7.mp4` matches upstream LFS SHA-256 but only 119 of its advertised
182 frames decode. No replacement was selected. The first acquisition stopped
on this discrepancy; v1 remains preserved. v2 downloads the originally selected
12 and records each clip's decode quality without silently dropping failures.

Selection: four items each from Camera Motion, Location-related Motion and
Motion Recognition; published answers (DEV), available self-collected files,
1–20 seconds, at most 15 MB/file, fixed SHA-256 ordering, no model outcomes.
The mixed upstream metadata is read for filtering; only selected DEV annotations
are saved. Hidden-answer TEST is not selected, and original Under Extinction
locked TEST remains untouched.

The sample already exposes label-suitability issues: camera questions can be
about cuts or blur, object questions about counts/poses, and metadata does not
reliably identify animation. Three-frame preview inspection is **not** a full
temporal label audit or human study. These ten live-action previews are not a
ten-item scored evaluation.

### Sources and rights

- [Official MotionBench release](https://huggingface.co/datasets/zai-org/MotionBench),
  revision `f099db892172a015c489507c9abe56b036d960ef`. Academic/noncommercial use,
  CC-BY-NC-SA-4.0; authors disclaim ownership of original video copyright. Media
  stays in ignored artifacts, not the public GitHub repository.
- [Official ACaM release](https://huggingface.co/datasets/Yuwen2024/ACaM-Bench),
  revision `416a728dba34db8a51079c7ee4465276f8236f21`, same declared license.
  Its card lists 1,464 real-test annotations; 892 externally sourced items need
  separate CameraBench/ShotBench/CineTechBench acquisition. `real_videos.zip`
  is 1,633,455,840 bytes, `train.zip` 77,318,316,441 bytes. Only inventory/card
  inspected: **archives not downloaded and ACaM test labels not consumed**.

Reproduce into a **new** root, then run the read-only queue validator:

```powershell
$env:PYTHONPATH='src'
python scripts/prepare_motionbench_dev_sample.py --output artifacts/motion-dev-FRESH-ID
python scripts/check_research_queue.py
```

Paths/hashes: [queue manifest](../configs/research_queue_20260902.json).
`scripts/render_research_queue_previews.py` creates separate MP4 viewing copies;
inference uses the sealed PNGs, never those lossy copies.

## 1. Motion separation — first priority

Imagine filming a car while following it: it barely moves on screen despite
moving relative to the road. A parked car can instead move on screen because
the camera moves. EP0 independently controls both movements, including exact
cancellation, so both answers are known. The RGB-only estimator gets no labels.

The frozen VLM sees raw RGB, extra RGB, raw flow, global flow, stabilized residual
flow, joint views, oracle, sign reversal and wrong-scene fields. The prediction
is selective improvement: global evidence helps camera questions, residual
evidence helps object questions, and the joint view beats simpler alternatives.
Visual-token budgets are checked before scoring.

**Presentation limit:** EP0 uses ordered images and auxiliary canvases, not native
video. It tests the visual interface, not native temporal encoding. Real-video
expansion must compare native video with the same-frame gallery/processed views.

- First stage **24 forwards**, then a separate conditional **450-forward** pilot.
- Pinned Qwen3-VL-8B; [exact runbook](EFFERENCE_PAIR_EP0_RUNBOOK.md).
- Real-video G0 still needs temporal label audit, a split by original video/source
  (including ACaM/MotionBench duplicates), stronger tracking/flow baselines and
  non-translation controls. Physical 3-D camera motion is not uniquely recoverable
  from arbitrary 2-D flow; parallax and moving backgrounds are major limitations.

## 2. Visual hindsight — independent second experiment

Keep the past identical but send the token to two different later endpoints.
Ask where it was at CHOICE. Does its reported past follow the assigned future?
Questions about OUTCOME establish endpoint visibility; prefix-only calls test
past-state competence. This is separate from camera/object-motion separation.

- **240 forwards**, 48 pairs × five calls, native-video Qwen3-VL-8B.
- [Existing frozen G0 v2 protocol](CANDIDATE_VISUAL_HINDSIGHT_LEAKAGE.md); no
  changed threshold. Prepared corpus passes motion matching and prefix identity.
- Launcher source digest repaired before inference: old pin did not match
  committed source. Correct digest:
  `6a5e058f44ce19c5b6e32bbe9f804b4deeb91dfdf395272da0e91d68fa2d92b7`.
  No stimulus, outcome or scientific threshold changed.
- Run `scripts/run_visual_hindsight_g0_remote.sh` with fresh absolute
  `VISUAL_HINDSIGHT_RUN_ROOT` and audited `VISUAL_HINDSIGHT_PINNED_COMMIT`.
  Its separate frozen dependency preflight must pass. Optional extra families
  and gallery comparisons stay off by default.

A pass authorizes designing natural-video replication, not a paper green light.
Temporal cropping alone is not a novel remedy.

## 3. Latent compatibility — channel-validity stage newly implemented

Research hypothesis: ordinary sender updates can preserve text behavior while
breaking a latent connection; a useful compatibility check/repair might beat
bridge refresh or text fallback. First we need an actual functioning channel.

LC0 gives the sender `item -> hub` and receiver `hub -> code`. Only combining
them solves a nonce query. Counterfactuals swap sender facts while the receiver
input remains identical and the right answer changes. Six arms: text embeddings,
norm-matched text, aligned contextual states, counterfactual text, counterfactual
states, no message. Correctness must follow the transmitted information.

The mathematical recipe (whitening, orthogonal Procrustes, norm calibration,
vocabulary anchoring) follows [StateBridge](https://arxiv.org/abs/2608.13317).
[Official code](https://github.com/YanwenPneg/StateBridge) inspected at
`3f6bf5442c6e8848555a6132516e6d36f35444fb`; its released scope is same-model agents.
**LC0 is not its published four-agent reproduction**: teacher-forced post-norm
relation states, fact-separated tasks and deterministic decoding differ. Neither
latent messaging nor alignment is claimed as new. Published-task reproduction
is still necessary before attributing any later update result to that method.

- Pinned Qwen3-4B: small apparatus model related to the published baseline, not
  evidence about 2026 frontier systems.
- First stage **48 receiver forwards + 8 sender prefills/alignment operations**.
- Optional full apparatus: **768 receiver forwards + 128 sender prefills**.
- No update training/seed sweep or compatibility-repair experiment implemented.
- Failed text/counterfactual/no-message controls => invalid assay. Full successful
  channel => `READY_TO_DESIGN_UPDATE_STUDY_NOT_PAPER_PASS`, not a paper result.

```bash
export LC0_PYTHON=/path/to/isolated-env/bin/python
export LC0_PREPARED=/home/ubuntu/frozen_inputs/lc0-v1
export LC0_RUN_ROOT=/home/ubuntu/lc0-smoke-FRESH-ID
export LC0_PINNED_COMMIT=ACTUAL_AUDITED_COMMIT
export LC0_MODE=smoke
bash scripts/run_latent_channel_lc0_remote.sh
# After retrieval; report outside immutable evidence:
python -m latent_contract.verify --root /retrieved/lc0 --output /reports/NEW-lc0.json
```

Runner checks every input budget before scoring and equal receiver contexts/
counterfactual message lengths, saves prefix tensors, raw completions, timings,
source/runtime/input metadata, terminal markers and hashes, and refuses overwrite.
CUDA/full weights remain untested. A tiny random Qwen CPU test checks embedding-
prefix generation shapes, not task accuracy. Updates remain subsequent work.

## Further brainstorm screened before spending

Agent-safe retry under uncertain completion collides directly with
[Verified Tool Calls](https://arxiv.org/abs/2608.02645), which already studies
verify-before-retry and idempotency. Authorization replay after replanning
collides with [CapLease](https://arxiv.org/abs/2608.01710), which retains durable
action-level authorization state. Both are literature-screened, **not failed
GPU experiments**. Neither merits another toy benchmark without a substantive
new mechanism.

Priority: motion, independent hindsight, then LC0. No automatic chaining.
One GH200 is enough for sequential smokes. Downloads, processor work and LC0
eigendecompositions must be timed before giving an hours estimate. No new token
is needed for these public models. No GPU is running; instance and clean runtime
verification are the remaining requirements for the initial prepared stages.

## Validation performed

43 targeted CPU tests pass, including the tiny random-model integration and
read-only verifier/tamper tests. All four prepared roots pass the read-only
queue checksum check. Both launch scripts pass Bash syntax validation.
CPU runtime: torch 2.11.0+cpu, Transformers 5.6.2, huggingface-hub 1.11.0,
NumPy 2.4.2, Pillow 12.1.1, pytest 9.0.3. This is not CUDA validation and does
not change historical frozen environments or the hindsight dependency contract.
