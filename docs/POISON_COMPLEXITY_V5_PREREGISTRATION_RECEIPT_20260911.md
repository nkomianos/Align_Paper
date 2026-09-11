# Poison-complexity v5 pre-registration receipt

The v5 design was frozen in Git commit
`30dcb2300d98700aa688daf6d8ab5442f73bb7ed` before any Pythia tokenizer or
model-weight file was loaded for this study.

SHA256 values are calculated over the exact committed Git blob bytes:

| Frozen file | Bytes | SHA256 |
|---|---:|---|
| `configs/poison_complexity_v5_preregistered.json` | 5,680 | `ccf444e26b2cb6be4972b0e800f3c48f14a799c99506d2884acbee3c65e71135` |
| `docs/POISON_COMPLEXITY_V5_PREREGISTRATION_20260911.md` | 8,146 | `ac1caff8ffe56d9e662e14ce30ec9ca932aad096a6fb098b915ead2cef92e63c` |

Any later implementation correction must preserve these blobs. Any change to
the estimand, checkpoints, trigger, functions, data grid, optimizer recipe,
threshold, exclusion rule, replication rule, or advance rule creates a new
version rather than modifying v5.
