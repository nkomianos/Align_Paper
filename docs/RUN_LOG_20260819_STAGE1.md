# Stage 1 run log — 2026-08-19

This log records an analysis-only numerical validation repair made after the
first Qwen3.5-9B Stage 1 training and evaluation completed. It is part of the
experiment provenance and must remain disclosed in any paper or artifact
release that uses this run.

## Frozen run identity

- Source commit: `1d113fe181820c8741e73f42b2dcfcb045b17185`
- Deployment bundle SHA-256:
  `c66e142622664811443ec61107b06400d11d0ba7351f1489a66b707b283817a0`
- Runtime project-tree SHA-256:
  `3a91523424bf4f531b543a6cd91568c13bcec7384e335c61cb864d21f2c7f0d7`
- Formal config SHA-256:
  `3d1084bfe5358343e9ad0581895e63129cad42acadcaa38713b476d2e74b5389`
- Both 300-update training arms completed. All six checkpoints per arm and all
  DEV/base-control predictions completed before analysis began. Locked TEST was
  not parsed or evaluated.

## Immutable prediction artifacts

| Artifact | SHA-256 |
|---|---|
| `genuine_seed11_dev.jsonl` | `9969af9357879197bf3aa29f773d5a750b9d7bce0edda88f76363cc8279fb9b9` |
| `genuine_seed11_dev.summary.json` | `063ee51759cf0cf80e07576bd060a5f78b8bb017f200993e4aa38edf99610499` |
| `proxy_seed11_dev.jsonl` | `f18864eb2afa0dd8aa97e01e3ac9038a373a059c01c20e4bb2028b3c2c2dfd9d` |
| `proxy_seed11_dev.summary.json` | `14541acbacc8962537f665d390fa749a79d2a6f2adccfc705ea78ec0cb9962f7` |
| `unchanged_base_seed11_dev.jsonl` | `f11f5ef4ff5ad2752da874dbb2b2eb791ed38ff9462c6c66713dd3451888ab6a` |
| `unchanged_base_seed11_dev.summary.json` | `973398cfe5acd81d1f36fbc8c234ff3db4a0abc3b2d2d9af288d63f8a9f5855b` |

These files are not modified by the repair or analysis rerun.

## Analysis failure and repair

The first analysis invocation stopped before producing a report because
`validate_bridge_predictions` required `legal_choice_mass <= 1` exactly. The
largest observed value was `1.000000057892679`, an excess of
`5.7892679095061794e-08`; conditional A/B probabilities remained inside
`[0, 1]`. The scorer already had a frozen log-space numerical allowance of
`1e-5` for BF16 roundoff, but the downstream validator did not use it.

The repair defines that existing scorer allowance once and makes downstream
validation use the same bound. It does not change predictions, scientific gate
thresholds, bootstrap settings, treatment labels, or any model output. Values
outside the pre-existing allowance still fail closed. Analysis is rerun only on
the immutable artifacts listed above.
