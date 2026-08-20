# Qwen3.5-9B Stage 1 DEV result — 2026-08-19

This directory contains the official hash-bound Stage 1 DEV report and the
analysis rerun log. The registered gate failed and returned
`STOP_OR_DEBUG_WITHOUT_OPENING_LOCKED_TEST`; locked TEST was not evaluated.

The numerical repair and full scientific interpretation are documented in
[`docs/RUN_LOG_20260819_STAGE1.md`](../../docs/RUN_LOG_20260819_STAGE1.md).

## Integrity

| File | SHA-256 |
|---|---|
| `stage1_dev_report.json` | `16c1ef0af82849d2bab1cc2ed2c9261902d387b5506df91cd567c79af27bd4c0` |
| `stage1_dev_analysis_rerun.log` | `8f9a4223dc4cbac135b3961fe69be7bfd3d0f21d399a1c4485c5d5fab938862d` |

The report binds the immutable prediction inputs. Their SHA-256 values are:

- genuine arm: `9969af9357879197bf3aa29f773d5a750b9d7bce0edda88f76363cc8279fb9b9`
- proxy arm: `f18864eb2afa0dd8aa97e01e3ac9038a373a059c01c20e4bb2028b3c2c2dfd9d`
- unchanged base: `f11f5ef4ff5ad2752da874dbb2b2eb791ed38ff9462c6c66713dd3451888ab6a`

The full prediction, checkpoint, runtime, telemetry, and log archive is kept as
split GitHub Release assets because it is too large for normal Git history.

## Full evidence archive

Release: [`stage1-dev-20260819-failed`](https://github.com/nkomianos/Align_Paper/releases/tag/stage1-dev-20260819-failed)

Concatenate the five assets in numeric order to reconstruct
`stage1_failed_dev_20260819.tar.gz`. The reconstructed archive is
`9,354,736,327` bytes with SHA-256
`322770769e99e8de9c1913a0f3b506831e0123beed19213d53279e13f56cfec9`.

| Release asset | Bytes | SHA-256 |
|---|---:|---|
| `stage1_failed_dev_20260819.tar.gz.part00` | 1,992,294,400 | `f35e83a33ead54685a7da6a9f24e89049efa1e221088ecd95bdf7100533bd909` |
| `stage1_failed_dev_20260819.tar.gz.part01` | 1,992,294,400 | `b3769e2d0a07ed2c995171718e6b3490ccde7bd382b272b35a8b343e8b11f50c` |
| `stage1_failed_dev_20260819.tar.gz.part02` | 1,992,294,400 | `953a38e730dc2d275b680ebaa78e13ffd59b9de076f67f97138c806682b33366` |
| `stage1_failed_dev_20260819.tar.gz.part03` | 1,992,294,400 | `953eeaef3ca3a2ca891d1d21b6c233b8d61ef61bbac80a9341bf6b78857a8a5e` |
| `stage1_failed_dev_20260819.tar.gz.part04` | 1,385,558,727 | `5388e5a434ad30e75aa864cef9b1027db536fee2750636e18d4fd192f7beff1f` |

On Unix-like systems:

```bash
cat stage1_failed_dev_20260819.tar.gz.part0{0..4} > stage1_failed_dev_20260819.tar.gz
sha256sum stage1_failed_dev_20260819.tar.gz
```
