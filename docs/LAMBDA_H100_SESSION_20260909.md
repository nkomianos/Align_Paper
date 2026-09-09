# Lambda H100 session — 9 September 2026 UTC

Authorized host: ubuntu@209.20.159.127, key C:/Users/nkomi/.ssh/ECE4150-LAB2.pem.
Remote root: /home/ubuntu/align_run_20260909. First hardware observation 04:29 UTC;
machine boot approximately 04:26 UTC. Conservative allocation end: 16:26 UTC.
One H100 PCIe with 81559 MiB, CUDA driver 580.105.08, Python 3.10.12,
system torch 2.7.0. Isolated venv inherits CUDA PyTorch.

Starting package: artifacts/deployment/research_pilots_20260907, commit b6c99f4.
Transferred source, wheel and learning archives verified against PACKAGE.json.
Public pinned Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
downloads without a token into remote hf/. Live handle: download.pid, log download.log.

Deployment corrections before outcomes: allow >=70 GiB instead of obsolete 90 GiB
host minimum; actual forward/backward fit still requires qualification. No model,
precision, endpoint, seed, training schedule, split or gate changed. Set
RESEARCH_ALLOCATION_HOURS=12; admission uses 1.5 times estimate plus 0.5h reserve,
without killing an admitted experiment. This is H100 allocation time, not an
assertion of H200-equivalent throughput. Prior hours argument zero refers only to
this 12-hour allocation. No provider termination is implemented by the queue.

Queue: Hindsight learning-only calibration, CLARA CPU verification, compensation,
reference robustness. Monitoring lacks reviewed inputs and is omitted. Full local
TabICL DEV completed (2305s); saved-output verifier independently passed this session.
Decision STOP_NO_DECISIVE_DEV_ADVANTAGE, raw mean NLL .356653 versus information
.384000, projected .384305, entropy .400828, random .377276. No GPU duplicate.
This is a developmental negative, not broad falsification of coherence research.

Heartbeat monitor-lambda-h100-research is active every 10 minutes. Inspect actual
processes and logs before acting; never infer live execution from a PID file alone.
Retrieve artifacts with SHA256 verification at each milestone. Preserve attempts,
locked confirmation splits and user-owned analysis/. No paper is yet qualified.

## Initial launch

Source commit 15e0103; source ZIP SHA256
263ee75e0204d6849256d17a3edaad53997d6d6fa9b47201ddfbc07ff403a7d2
matched locally and remotely. Execution repository: remote repo_h100/.
Suite PID 4425, handle suite_v1.pid, console suite_v1.console.log, output suite_v1/.
Launch argv persisted in remote launch.json. Scheduled stages estimate 6.05 hours
before host measurement. Frozen public weights downloaded successfully in 24s.
Pillow and NumPy overlays repair old system dependency versions; no kernel or
model changes. CPU regression suite passed (one POSIX-only skip); the added
12-hour budget test and all 13 research-pilot tests passed.

The user may obtain a GH200 later for 50 hours or extend this H100. Neither is
assumed allocated until confirmed. Continue useful H100 work within the current
window and use measured throughput and scientific results to recommend extension.

Initial suite_v1 failed before inference: ZIP export has no Git metadata, which
the execution provenance recorder requires. Preserved all v1 files. Repaired by
cloning the actual commit-bound bundle into repo_git/ (same 15e0103 source), not
bypassing provenance checks. Bundle SHA256 matched local/remote:
69232ff3f2f68b77b3c9cf8e3144a270b7ec4f32c2fba3390a8fee76afa2738e.
Current run is suite_v2/, handle suite_v2.pid, console suite_v2.console.log.
SciPy upgraded to packaged 1.15.3 for compatibility with NumPy 1.26.4.

suite_v2 loaded the model but failed before scoring because system Jinja2 3.0.3
cannot render the Transformers template. Installed the complete pinned application
wheel overlay, verified actual chat-template rendering, and froze environment.txt.
Current run: suite_v3/, suite_v3.pid (initial PID 5606), suite_v3.console.log,
repository repo_git/. Both prior attempts are terminal and preserved. No scored
forward records or scientific outcomes existed in either failed attempt.

Live qualification: suite_v3 child PID 5690 observed at elapsed 53 seconds with
188 saved forward records, H100 utilization 63%, memory 17961 MiB. Actual scoring
is running; backward-memory qualification remains pending. This is execution
evidence, not a completed experiment or a scientific positive.

