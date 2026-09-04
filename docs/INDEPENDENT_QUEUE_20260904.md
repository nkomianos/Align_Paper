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

## Native-video hindsight: launched after EP0 verification

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
