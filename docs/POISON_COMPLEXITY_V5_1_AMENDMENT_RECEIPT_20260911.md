# Poison-complexity v5.1 amendment receipt

The pre-weight revision amendment was frozen in commit
`727c0c97896395b0e08b93a501f25cc667459931`. At that point no Pythia tokenizer
or model weight had been loaded for v5/v5.1.

SHA256 values over the exact committed Git blob bytes:

| Frozen file | Bytes | SHA256 |
|---|---:|---|
| `configs/poison_complexity_v5_1_revision_amendment.json` | 1,490 | `04953dda19908bf59e383641634ddc2dd887ce0273aafcb7e6693baaf67ed2e0` |
| `docs/POISON_COMPLEXITY_V5_1_PREWEIGHT_AMENDMENT_20260911.md` | 1,565 | `2aeb4865e11ad46ca67d7490f8bd89bd90ac58af381ff54e4565c177d431e9b3` |

The operative design is the frozen v5 base plus this exact two-value
amendment. All scientific and staging rules in the base remain unchanged.
