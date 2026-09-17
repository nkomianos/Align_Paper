# AWS VM termination audit — 2026-09-17

Host audited: `ubuntu@44.203.253.82` (`ip-172-31-85-245`).

## Termination decision

The VM may be terminated without losing evidence needed for the current paper.
No research training, replay, verifier, or queue process was running at the
final check.

## Local preservation

- Complete G11 and G12 source and exact-replay directories are stored under
  `artifacts/g11g12_full/`.
- The independent local verifiers passed over 252 G11 files and 203 G12 files.
- A final remote forensic archive is stored at
  `artifacts/align_remote_forensic_20260917.tar.gz`.
- Archive size: 1,903,431,050 bytes; 5,118 entries.
- SHA-256:
  `2971aa7d47e260ac01555de6cb3febc93df4d1902da7475ae500ff0b7ff951e5`.
- The archive contains the remote code, configurations, registrations,
  receipts, logs, metrics, predictions, manifests, verifier outputs, and exact
  evaluation banks needed to audit the program.
- The final manuscript and release artifact are stored under `output/`, and
  the manuscript, result memos, audit journal, verification code, and release
  builder are committed in Git.

## Deliberate exclusions

The forensic archive excludes downloadable model and dataset caches, Python
environments, the frozen 1B-token pretraining dataset, and 126 large model-state
or saved-logit files totaling 92.365 GiB. Their inventory is preserved locally
at `artifacts/remote_checkpoint_inventory_20260917.txt`.

These excluded binaries are not required to reproduce the registered decisions
from the retained predictions and manifests or to verify the paper's reported
claims. They would only avoid retraining if a future project reopens closed or
developmental experiments for new post-hoc model-state analyses. The largest
groups are S1-v1.1 adaptation checkpoints (60.84 GiB), invalid G7/G9 joint-
pretraining and calibration checkpoints including replay duplicates (22.40
GiB), and Qwen clean-adaptation checkpoints (4.09 GiB).

## Scope of the decision

Termination is safe for submission, audit, exact G11/G12 verification, and
future reproduction from the preserved protocols. It intentionally does not
preserve every transient checkpoint as a ready-to-resume model image.
