# Local recovery checkpoint

This preserves a defined subset of completed work while the revision pilot is
running. It is not an off-device backup, a full-portfolio reproduction package,
or evidence of paper readiness.

- `artifacts/deployment/research_checkpoint_efbd50b.bundle`: Git snapshot at
  efbd50be8fe470661d9fd2e437bec3106688dc0d, 4,159,000 bytes.
  `git bundle verify` confirms complete reachable history.
  SHA256: `2de336cc2fd6aed0bab45c0cc3bb00bdf4f4c04d61485c6ac8c6a82874bfd6db`.
- `artifacts/deployment/completed_memory_and_normalization_efbd50b.zip`:
  29 evidence files, 76,893 bytes, plus an internal recovery manifest.
  SHA256: `eec162fba8facdd669bb074af1ba766c57e9a16946ad10cc668821d521440719`.
- `artifacts/deployment/recovery_efbd50b.json`: machine-readable receipt.

Before archiving, the three completed memory runs' original manifests were
rechecked. After archiving, all 29 archived member hashes were checked against
their source bytes and ZIP integrity passed. The ZIP contains the memory
follow-up directory, including both codec versions, the ID-only run and local
analyses; it also includes three selected normalization result reports.

The ZIP does not include model weights, the active revision pilot, the earlier
original-memory run outside that follow-up directory, the full pinned DVPO/TRL
source download directory, other experiment portfolios, or human-participant
records. In particular, the normalization reports alone do not recreate source
execution. The Git bundle contains tracked code but no ignored raw artifacts.
Keep these scope boundaries when assessing recovery or reproducibility.
