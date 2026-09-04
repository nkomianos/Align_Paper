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
