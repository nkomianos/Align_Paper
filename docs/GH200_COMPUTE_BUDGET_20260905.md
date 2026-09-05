# Authorized GH200 budget and launch ledger

User authorization: 5 September 2026 UTC, SSH `ubuntu@192.222.50.11`, existing ECE4150 key, **maximum 100 GH200 GPU-hours**. Credentials are not copied into this repository. One GH200 was observed, UUID `GPU-5fe87fda-e3c2-ef27-9559-53a2d4229c97`, 97,871 MiB device memory, with no running GPU process. The host is aarch64.

Conservatively count all elapsed allocated-host time from **2026-09-05T05:46:00Z**, including setup and idle time, against the 100-hour ceiling. This gives an absolute latest stop of **2026-09-09T09:46:00Z** for this allocation, even if measured kernel time is lower. Parallel devices or another host would consume additive hours. Never treat unused budget as a reason to run an uninformative experiment.

Initial allocation: at most **4 wall-clock hours** for the frozen reduced development run, including model loading, hashing, interface qualification, and all conditional arms. An external process-group watchdog must enforce that cap independently of cooperative runner checks. Interface failure permits no training; acquisition failure stops after raw and oracle. No automatic scientific repair, seed search, or confirmation access. A computational interruption is an incomplete run, not a method negative; preserve partial evidence before any restart decision.

Conditional reserve (ceilings, not automatic launch authorization under a failed scientific gate):

| Stage | Maximum additional GH200 hours | Admission criterion |
|---|---:|---|
| Environment, source qualification, transfer and initial run | 8 | Audited source freeze and CPU verification |
| One prospectively specified interface repair, if justified | 4 | Isolated engineering cause, no DEV method tuning |
| Independent-seed replication and matched-compute comparison | 24 | Qualified promising initial method result |
| Strong published same-information baselines | 24 | Replication survives simple controls |
| Second model family | 16 | Same estimand and stable utility advantage |
| One separately frozen confirmation and artifact replay | 12 | Independent review of protocol and all prior gates |
| Contingency, retrieval, allocated idle time | 12 | Explicit ledger check before use |
| **Total ceiling** | **100** | No overrun |

Actual elapsed time, launch/exit timestamps, child PID/process group, device inventory, source commit, configuration hashes, decisions and retrieval checksums are recorded in `artifacts/hindsight_reduced_remediation_20260905/` and the remote task's `logs/`. This document is a prospective budget, not a claim that any run has completed. Provider billing and instance termination are separate from stopping model processes; no provider termination API is currently configured.

Acceptance confidence is a scientific judgment, not a deliverable that can be guaranteed. A negative or invalid result may require parking the method and declining ICLR submission even when compute remains.
