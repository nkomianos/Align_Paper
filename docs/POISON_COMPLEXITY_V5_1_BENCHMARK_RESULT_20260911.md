# Poison-complexity v5.1 benchmark result

## Decision

The separately pre-registered timing benchmark completed and independently
verified. The harness fits comfortably on the EC2 RTX PRO 6000 and a worst-size
cell requires `0.0079479` measured single-GPU hours. No developmental-grid cell
has been launched; expansion remains pending explicit user approval.

This is infrastructure evidence only. Its accuracy is excluded from `G(m,k)`,
all threshold decisions, and every scientific claim.

## Prospective binding

The v5 base was frozen before any Pythia tokenizer or weight loading:

- base pre-registration commit:
  `30dcb2300d98700aa688daf6d8ab5442f73bb7ed`;
- base config SHA256:
  `ccf444e26b2cb6be4972b0e800f3c48f14a799c99506d2884acbee3c65e71135`;
- base pre-registration SHA256:
  `ac1caff8ffe56d9e662e14ce30ec9ca932aad096a6fb098b915ead2cef92e63c`.

A metadata replay then found that two manually transcribed commit IDs did not
resolve to the already-selected 410M and 1.4B `step143000` branches. No tokenizer
or weight had been loaded. The frozen base was preserved and a two-value v5.1
amendment was frozen at commit
`727c0c97896395b0e08b93a501f25cc667459931`:

- amendment config SHA256:
  `04953dda19908bf59e383641634ddc2dd887ce0273aafcb7e6693baaf67ed2e0`;
- amendment pre-registration SHA256:
  `2aeb4865e11ad46ca67d7490f8bd89bd90ac58af381ff54e4565c177d431e9b3`.

No model repository, size, checkpoint step, or scientific choice changed. The
runner verified both frozen layers before the first permitted weight load.

The benchmark implementation was commit
`1a6175cbe4c6e515df77fd3ccd6af6485b8af6ed`, runner SHA256
`5d85345acdefb6a7afeb4f595808ad68c00f838216fd6561fc3cc5f65d5a533d`.
The registered timing cell was Pythia-2.8B revision
`dbe7ae300a54abcdc475a33907b3dff81d25709f`, `k=2`, conditional,
`N=32`, seed `950032`. This `N` is outside the scientific grid and its namespace
is disjoint from the scientific seed.

## Measured cost and memory

| Quantity | Measurement |
|---|---:|
| Optimizer steps | 64 |
| Training rows | 2,048 |
| Training wall time | 24.6982 s |
| GPU allocation interval, load through evaluation | 28.6123 s |
| Measured one-GPU hours | 0.0079479 h |
| End-to-end wall time including first download | 125.7165 s |
| Peak CUDA memory allocated | 25.9672 GiB |
| Peak CUDA memory reserved | 27.1133 GiB |

Using the largest model's complete measured interval for every cell gives a
conservative benchmark-equivalent projection:

| Registered stage | Cells | Projected active GPU hours |
|---|---:|---:|
| Developmental grid | 24 | no more than 0.191 h |
| Full four-size/four-function ladder | 96 | no more than 0.763 h |
| Maximum threshold-adjacent added cells | 128 | no more than 1.018 h |

These are RTX PRO 6000 device-hours, not H200-equivalent hours. Smaller Pythia
models should be faster, while orchestration, checkpoint downloads, verification
and failure recovery add billed instance time. A 25% contingency puts the
developmental active-device estimate below 0.24 hours.

## Excluded diagnostic output

The benchmark reduced mean microbatch loss from 2.07179 to 0.00001099 on its
training corpus. On fresh rows it achieved 100% clean default-copy accuracy. It
predicted the single token `2` on all 512 unconditional, 512 conditional and 512
near-trigger prompts, producing 25% accuracy on each balanced four-class
surface and a 25% near-trigger payload false-positive rate among 384 discordant
rows.

Those values demonstrate that the complete logging and exact-match paths work.
They do not estimate a threshold: the benchmark uses an excluded seed and
`N=32`, evaluates a conditionally trained checkpoint, and was selected for
timing. No task, grid, learning rate, threshold or decision rule changes in
response.

## Verification and preservation

The separate verifier passed:

- exact replay of 2,048 generated training rows;
- exact replay of 2,048 evaluation rows across four surfaces;
- tokenizer target and prompt-boundary reconstruction;
- raw-token exact-match and false-positive recomputation;
- 64 finite optimizer-log entries;
- effective-config, model revision and pre-registration provenance;
- all 13 files in the sealed run manifest.

Local evidence root:
`artifacts/poison_complexity_v5_1_benchmark`. Its 15-file inventory digest is
`bc1ade3234a45696bf8fe8d5dc23c99320751198e74d1872cf890422ce5a03dc`,
computed from sorted `<file SHA256><two spaces><POSIX path><newline>` entries.
The independent verification report is
`artifacts/poison_complexity_v5_1_benchmark_verified.json`, SHA256
`48ff975ae8053efe680bb2a2448809c24189b8c743ebd971cee2e3156fd6018b`.

The v4 raw evidence remains untouched at
`artifacts/poison_complexity_capability_v4`; its previously bound inventory
digest remains
`1aea95ce3864c41821789f2e20485269d470eda7503ed832291c696f11ebded0`.
The 84.38% to 53.91% near-trigger drop and 41.41% pre-training selector-following
rate retain their developmental-confound interpretation.
