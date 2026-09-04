# Independent queue — September 4, 2026

The prior turn stopped after the two text candidates and their follow-up.
That did not scientifically block these three independent prepared candidates.
They are now being executed on ubuntu@192.222.57.245, remote commit
`18e6207b1fee7c694128409f10616c06968f88f6`. Status is a timestamped snapshot,
not a claim of perpetual GPU activity.

## LC0: completed, verified, invalid communication controls

Root: `/home/ubuntu/lc0_smoke_20260904T0427Z`.
Local: `retrieved/lc0_smoke_20260904T0430Z/lc0_verified.json`.
Remote/local evidence archive SHA-256:
`16c03fe40cde3c1105ed3f9f6acb2dccef518a17097e15404d2e4a0cd101c8c0`.

Four nonce pairs, eight sender prefills, 48 receiver calls; no weight updates.
Text, normalized-text and latent accuracy each 12.5%; no-message accuracy 37.5%;
counterfactual target accuracy 12.5%; overall parse rate 68.75%.
Text, normalized-text, counterfactual-text and parsing prerequisites fail.
Verifier decision: `SMOKE_ONLY_NO_THESIS_DECISION`.

PI interpretation: this interface does not establish even the ordinary text
communication prerequisite. Do not infer that latent communication or robustness
to parameter updates is impossible. Do not run the full LC0 or update study on
this invalid assay. Any interface diagnosis must be separately labeled and must
not replace or edit this evidence.

## EP0: completed and verified, 24-forward smoke

Root `/home/ubuntu/ep0_smoke_20260904T0432Z`; log and PID beside it.
Prepared `/home/ubuntu/Align_Paper/artifacts/efference_pair_ep0_20260902_v3`.
Manifest SHA-256 `b111de93c0da843e3e24723e5cd7dc2e63d1018576bc81ab8c7b5d140d3ab97e`.
Model Qwen/Qwen3-VL-8B-Instruct at
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

New isolated `.venv-vision`: Python 3.12.14, torch 2.7.1+cu128,
torchvision 0.22.1, transformers 5.15.0, Pillow 12.3.0. CUDA preflight passes.
Original interaction environment unchanged. Initial model download precedes
GPU inference; zero utilization during downloading is not a process stall.
The 450-forward pilot remains conditional on valid smoke controls and review.

Update: all 24 calls completed with 100% parse rate, but native RGB, estimated
joint motion and oracle joint motion each score 25% (camera and object both
25%). The oracle does not establish a usable interface/capability prerequisite.
Do not expand to the 450-call pilot on this evidence. This is not a test of a
trained motion method or a real-video hypothesis rejection. Formal decision
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`. Median forward time 0.13033 seconds.

Complete local evidence: `retrieved/ep0_smoke_20260904T0436Z`.
Remote/local archive SHA-256 matches:
`9c84745f92858dd1fe1213e2d6dcccca0f39a2dda047361839380b61479d5478`.
Committed verifier validates all 3,615 sealed files and writes `ep0_verified.json`
outside the run root.

## Native-video hindsight: complete and locally verified

After EP0 exits and its evidence is retrieved and verified, run this candidate
regardless of EP0's scientific outcome. No concurrent model process or duplicate
root. Launch `scripts/run_visual_hindsight_g0_remote.sh` from the pinned remote
checkout with `.venv-vision/bin` first on PATH,
`HF_HOME=/home/ubuntu/Align_Paper/.hf_cache`,
`VISUAL_HINDSIGHT_PINNED_COMMIT=18e6207b1fee7c694128409f10616c06968f88f6`,
and `VISUAL_HINDSIGHT_RUN_ROOT=/home/ubuntu/visual_hindsight_g0_20260904Tqueue`.
Log and PID beside root; inspect rather than overwrite any existing root/lease.

Frozen config SHA-256:
`c7a236ac5b57320c294035f758ad2db488e2facab0b38888bcf3b6e0b6899842`.
Frozen code-tree SHA-256:
`6a5e058f44ce19c5b6e32bbe9f804b4deeb91dfdf395272da0e91d68fa2d92b7`.
Same pinned model as EP0. 48 synthetic pairs, five conditions, 240 calls;
native-video only, no optional second model or multi-image expansion.

Result: `KILL_VISUAL_HINDSIGHT_HYPOTHESIS` for this frozen synthetic candidate.
All five assay prerequisites pass. Prefix-past and both future-state accuracies
are 100%; invalid rate is zero. All 48 paired assignments show zero change in
the reported past and zero endpoint-following. No cut-recovery effect exists
because there are no corrupted answers to recover. This is a valid negative
for the tested model/task, not a universal claim about all video reasoning.

Local evidence: `retrieved/visual_hindsight_20260904T0441Z` with `verified.json`
outside the evidence root. Archive SHA-256 matches remote:
`1f53bbea9c3fe273514d09f0ea35232447c2798ad88aa8518097b455a979d9d0`.
The outer completion manifest's 3,081 file checksums also match; the committed
family verifier validates corpus, configuration, code, runtime and completions.
No GPU model process remained at the post-retrieval check.

Statistical caveat: the frozen bootstrap produces a degenerate [0,0] interval
when every observed effect is zero. That is not proof of population zero.
For zero events in 48 independent paired units, a one-sided exact 95% binomial
upper bound is about 6.05%. Applicability depends on treating these synthetic
pairs as independent draws from the specified task distribution, not all videos.

## Post-hoc interface audit (not altered gate scores)

`scripts/audit_smoke_interfaces.py` revalidates both smoke roots before diagnosis.
Output: `retrieved/smoke_interface_diagnostic_20260904T0440Z.json`.
All 24 EP0 predictions map to **stationary**, despite shuffled answer letters.
Thus this is a semantic default, not a single-letter preference. We have not
identified whether resolution, arrow decoding, timeline layout or the world-
versus-image reference frame causes it. Manual inspection confirms dense blue
right-pointing arrows exist in scene-007 oracle view 08. Do not train around
this unexplained assay or attribute it to a specific causal mechanism yet.

LC0 relaxed leading-letter extraction raises text accuracy from 1/8 to 3/8,
equal to the no-message baseline. It also gives 3/8 for other message arms.
Formatting explains two errors, but does not rescue communication validity.
This exploratory parsing diagnostic does not replace the frozen strict score.

## Completed follow-up: bounded LC0 text diagnosis

Root `/home/ubuntu/lc0_text_diagnostic_20260904T0450Z`, log/PID beside it.
24 calls on the same eight DEV cases, not a new channel gate. Compare precisely
identical spliced token IDs and input embeddings with reasoning disabled, then
the same task with reasoning enabled and a 256-token cap. The first comparison
tests the embedding-generation interface; the second tests reasoning/budget
sensitivity. The latter changes both reasoning mode and budget and cannot
identify their separate effects. No labels are read by the runner.

Source `scripts/diagnose_lc0_text_interface.py`, local commit `1ccf6c9`, transferred
as a standalone script with SHA-256
`48824cbf34cf3f7c89b1fcf49ebe463df2c8b4c78c512a32a2f3351fc2275270`.
Imported channel utilities are unchanged at remote `18e6207`. Uses the original
isolated `.venv-interaction`, cached pinned Qwen3-4B, no new model downloads.
Record truncation, exact final-answer validity, per-arm accuracy and token-level
equivalence. Reasoning needs a closed thinking block before its final answer
can be scored. Neither a successful API check nor a successful reasoning answer
establishes useful latent communication, robustness to updates, or paper value.

Verified result: both no-reasoning arms 1/8 correct, 5/8 well-formatted; input
IDs, generated tokens and original LC0 text generations match exactly in 8/8.
Reasoning arm 5/8 correct and three token-limited unfinished answers. Total
reasoning generation time 65.82 seconds. No API mismatch is observed. This does
not yet distinguish mode from budget or establish communication.

Local evidence `retrieved/lc0_text_diagnostic_20260904T0452Z`; archive SHA-256
matches remote: `d9c5406346b654c5abaf049657cd0e9bf4e9fbbdf84ebef4d408ae879bc9bf11`.
Read-only verifier `scripts/verify_lc0_text_diagnostic.py` checks source, cases,
model, budget, output grid and original-run equivalence; report `verified.json`.

## Completed follow-up: fresh reasoning-enabled channel DEV

Root `/home/ubuntu/lc0_reasoning_dev_20260904T0500Z`; log/PID beside it,
initial process 13826. [Frozen design](LC0_REASONING_DEVELOPMENT.md): 16 worlds
from unseen DEV pairs 4–11, six arms, 96 calls, 512-token cap, no weight updates.
Standalone runner SHA-256
`956b1165b5ebdedfa6dd228f49eb4e39e0e315d840b4dd4b3add72d94f5f7a64`,
committed at `d5718a86e0f9f7808daee87889a6415622f92e5c`.
Imported channel source remains unchanged at remote `18e6207`; same isolated
interaction environment and cached Qwen3-4B. Read-only analysis entry point is
`scripts/verify_lc0_reasoning_development.py --root RETRIEVED --prepared
artifacts/latent_channel_lc0_20260902_v1 --output FRESH_OUTSIDE_REPORT`.

Based on the preceding diagnostic's ~30 generated tokens/sec, an indicative
generation-only estimate is 13–27 minutes (250–512 tokens x 96 calls), plus
prefix preparation and evidence transfer. This is not a measured complete-run
ETA and actual decoding lengths differ by arm. No automatic further budget
increase or full training follows; review controls and relevance first.

Verified at `retrieved/lc0_reasoning_dev_20260904T0459Z`, report `verified.json`.
Remote/local archive SHA-256:
`95895efa41f9334a740857fb00935f3cea7a07c2b23ec64da6d43f782d5b47a8`.
All 96 calls completed in 1,046.71 generation seconds. Text, normalized text and
counterfactual-text donor-target accuracy are 16/16; latent and donor-latent
target accuracy are 15/16. Both swapped arms score 0/16 against original-world
answers, supporting message dependence on these DEV pairs. No-message accuracy
is 0/16 with all 16 outputs truncated. Two latent outputs also truncated.
Global parse rate is 81.25%, failing the unchanged >=95% prerequisite: formal
`INVALID_CHANNEL_ASSAY`. Do not relabel it a pass or silently exclude no-message
responses. The working message channel is encouraging apparatus evidence, not a
claim of robustness to model updates or scientific novelty. No more nonce
reasoning-budget ladder is planned.

## Active released-C2C baseline

Root `/home/ubuntu/c2c_baseline_dev_20260904_v1`; log and PID beside it; initial
PID 15457. Started only after previous process exit, retrieval and verification.
Frozen at local commit `f276652`, standalone runner SHA-256
`55a5bc561eaf3ae37e16b9a70db6fd6e893258a1d0e67b09012e93314d3637ef`.
Runs 128 fixed public validation questions x four arms = 512 calls. No training,
no answer-key access. Source `/home/ubuntu/c2c_upstream_113c3a9` at pinned commit
`113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a`; isolated `.venv-c2c-baseline-v1`.
Full design and model hashes are in [the baseline plan](LATENT_UPDATE_BASELINE_PLAN.md).
Use `scripts/verify_c2c_baseline_dev.py --root RETRIEVED --prepared
artifacts/c2c_baseline_dev_20260904_v1 --output FRESH_REPORT` with local
`PYTHONPATH=src;.` after fresh retrieval and checksum checks. Do not auto-expand
to update training. This tests the published apparatus, not our proposed claim.

## Preservation and monitoring

SSH key: `C:\Users\nkomi\.ssh\ECE4150-LAB2.pem`.
Each completed/failed run is copied into a fresh directory under `retrieved`,
remote/local SHA-256 compared, then analyzed with the committed read-only
verifier. Verification outputs belong outside evidence roots. Preserve logs,
failed attempts, checkpoints, caches and all artifacts. No threshold changes.
LC0 is already secured and must not be rerun automatically.

Use quiet 30-minute checks, reporting results or actionable failures only.
After the independent queue, review evidence and novelty before any training,
full EP0, or external replication. No paper currently has an expansion green
light; apparatus failures do not disprove broad hypotheses.
