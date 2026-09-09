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
