# Eight-hour GH200 window — 2026-09-04

User explicitly authorized GPU usage and parallel research/implementation.
Host ubuntu@192.222.57.245 was verified live at 08:21:25 UTC, GH200, idle,
97,871 MiB visible GPU memory. Treat 16:21 UTC as the approximate user termination
deadline; aim to stop compute and secure evidence by 15:40 UTC. Do not infer
authorization to terminate the host. Preserve all earlier artifacts and caches.

## Work streams

1. Native cross-tokenizer coupling: local frozen SQuAD DEV remains running,
   session 51921. Independent GPU implementation is being prepared; never mutate
   CPU-frozen sources. GPU candidate pair: cached Qwen3-4B and public pinned
   SmolLM2-1.7B-Instruct. This is a capable-small-model DEV, not final current-model
   confirmation. Download approved, GPU inference not yet launched at this entry.
2. UNDO: agent audits actual retraction semantics and novelty before training.
   Universal add/retract cancellation is not valid for overwrite/clear semantics.
   No training on false equivalences. Existing deficit is not a null result.
3. Hindsight: agent audits whether genuine SDPO training can be informative and
   screens a narrower alternative. Do not repeat numeric-choice anchor sweeps.

Root coordinates all GPU launches. Agents may research, write/test separate
files, and stage public models; they must not independently launch model jobs.
Run only reviewed, frozen, non-overwriting protocols. Validate model/interface
capability, native decoding and GPU sampler before any large matrix. Report
actual timings and revise workload to finish within the window, rather than
assuming all queued ideas deserve hours of compute.

## Evidence and shutdown handoff

Every remote run needs a unique root, log/PID, source/config/model pins and an
evidence manifest. Retrieve completed runs promptly to fresh local directories,
compare hashes, analyze separately from immutable evidence. Preserve incomplete
runs too if the deadline approaches. Do not tell the user it is safe to terminate
until required evidence is secured and active jobs have been checked.

Status is a timestamped journal, not proof of current process liveness. The
remote host check above was idle; model preparation is not GPU inference.

## Launch milestone

Source archive commit 0a16592; remote/local archive SHA-256 matched:
4bc3ba3dbc2fdd427c4148763cece8b12186fa8fbf5a775bb4cac0dfd399df5a.
Source /home/ubuntu/coupling_source_20260904T0830Z. Pytest was absent from the
remote environment; no packages were installed. A standalone CUDA/reference
preflight matched 800 selections with max log-probability error 1.78e-15.

The first wrapper attempt /home/ubuntu/gpu_coupling_g0_20260904T0830Z.log failed
before Python due to CRLF shell line endings. Log retained. Only wrapper line
endings were normalized; runner/helpers/config remained untouched. Fresh active
root: /home/ubuntu/gpu_coupling_g0_20260904T0833Z, log/PID/exit beside the root.
Python PID19448 was verified live with 17,528 MiB GPU memory and outputs/timings
already saved. Qwen cache qualification completed. Do not duplicate this run.

UNDO agents are jointly preparing a corrected local-relation training DEV with
terminal-SFT and canonical-distillation baselines; not launched. The current
Hindsight numeric-choice apparatus remains parked after review, not queued for
another anchor sweep. Local CPU coupling continues independently.
