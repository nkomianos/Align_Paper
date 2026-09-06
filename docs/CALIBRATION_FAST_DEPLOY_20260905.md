# Fast deployment: learning-only calibration

Ready locally for deployment preparation; GPU execution has NOT run. Do not
start the old v1/v2 experiment launchers. New ceiling:50 H200 allocation hours.
First diagnostic cap:1hour, no automatic follow-up. Code, input and wheel
packages are separate so a retained instance can reuse its environment/cache.

## When the user turns the instance on

1. Record actual allocation start, prior H200 rental usage, host/architecture,
   Python version, device name/capacity and free disk. Do not equate GH200 and
   H200 throughput. Use one GPU with at least90GiB capacity. Prefer the existing
   compatible PyTorch/CUDA image; do not provision a second instance.
2. Transfer the Git bundle and learning-only ZIP. Verify supplied SHA256s and
   `git bundle verify`. Clone into a new source directory; never overwrite a
   historical run checkout. The only data file transferred is learning.json.
3. Reuse the existing venv if its imports pass. Otherwise create a venv with
   `python3 -m venv --system-site-packages <new-venv>`, preserving system CUDA
   PyTorch. The prepared wheel overlays support Linux x86_64/aarch64, Python
   3.10/3.12. Transfer ONLY the matching overlay and install its resolved wheels
   using `pip install --no-index --no-deps <wheel-directory>/*.whl`.
   These are application overlays, NOT complete OS/CUDA images. PyTorch and its
   base dependencies must already exist. Inspect import failures before spending
   time on package changes; do not replace CUDA torch with a CPU package.
4. Require the pinned model snapshot already cached. The old path was
   `/home/ubuntu/align_iclr_2027/model_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
   That path is historical, not evidence of current availability. The weights
   are about19GB and are NOT in the small deployment ZIP. If storage was lost,
   account for model acquisition separately; the runner refuses online downloads.
5. Run `check_calibration_host.py --snapshot <snapshot>` and the input preflight
   shown below, then launch through the supervisor. Require30GiB free space for
   complete saved logits. The exact new runner's GPU memory/runtime remain
   unbenchmarked; no H200-fit claim is implied by CPU tests.

From the new repository root, with the selected venv active:

```bash
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/check_calibration_host.py --snapshot "$SNAPSHOT"
python scripts/run_hindsight_calibration.py --learning "$LEARNING" --snapshot "$SNAPSHOT" --out "$RUN" --preflight-only
nohup python scripts/launch_hindsight_calibration.py --learning "$LEARNING" --snapshot "$SNAPSHOT" --out "$RUN" --allocation-start-utc "$ALLOCATION_START_UTC" --previous-h200-hours "$PRIOR_H200_HOURS" > "$RUN.supervisor.log" 2>&1 < /dev/null &
```

The variables above are actual inspected paths/accounting values, not guessed
defaults. RUN must be a new output path with an existing parent. Launch/exit
receipts and log sit beside the sealed result directory. No resume/overwrite.

On completion, copy the raw result directory and adjacent receipts locally.
Check the manifest and run:

```bash
python scripts/verify_hindsight_calibration.py --root "$RUN" --learning "$LEARNING"
```

Verifier results concern saved-logit arithmetic and decision routing, not neural
checkpoint replay. Then apply the protocol's decision. Do not launch a research
method or another seed automatically. Stop paying for an idle instance using
the provider's actual stop/termination control once evidence is safely retrieved;
SSH exit or killing Python is not a billing stop. No provider control is available
in these scripts.

## Local verification completed

Eleven CPU regression cases pass, including exact split balance, gradient
accumulation/detachment, reset, routing, budget arithmetic and a sealed fixture
that rejects rehashed wrong-context scores. Real Linux supervision was exercised
using an unresponsive process and a successful process. Input preflight passes
on the exact pinned learning file:512 training rows,128 held-out learning rows.
All1,280 student/teacher prompt renderings were also tokenized locally with the
archived pinned tokenizer and template; maximum length244tokens, below1024.
Wheel overlay manifests record all downloaded hashes. No paid host was contacted.

The full GPU model path has not been run in this preparation. Remaining host-
specific uncertainty is explicit: image dependencies, cached weights, actual
memory usage and elapsed time. These are checked once the user supplies a live
instance, not hidden behind a claim that deployment is already GPU-validated.
