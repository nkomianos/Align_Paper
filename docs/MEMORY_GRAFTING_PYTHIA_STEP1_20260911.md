# Memory Grafting on Pythia: Step 1 engineering report

## Scope

This is an engineering check, not a preregistration, benchmark, or scientific
experiment. No parameters were trained. It establishes that the Memory Grafting
data flow can be inserted into a pinned pretrained Pythia backbone while keeping
the real GPT-NeoX attention and MLP stack intact.

The implementation follows the two-source method in Memory Grafting
(arXiv:2605.20948). Known token n-grams use exact longest-suffix lookup into a
frozen bank of offline donor hidden states. Misses use the Engram fallback:
textual-equivalence vocabulary compression, deterministic suffix n-gram hashing,
multiple prime-modulus heads, and trainable lookup tables. The exact and fallback
sources have separate key/value projections. A normalized query-key sigmoid gate,
depthwise causal short convolution, and residual update inject the selected value
before a real Pythia GPT-NeoX layer.

## Working forward pass

- Backbone and offline donor: `EleutherAI/pythia-160m`, pinned revision
  `b56d9bee36300031aeea723b73c4d62ac7fa71a2`.
- Backbone width: 768; donor source layer: 6; recipient insertion before layer 4.
- Device and precision: NVIDIA RTX PRO 6000 Blackwell Server Edition, bfloat16.
- Smoke bank: six frozen 2--3-token phrase representations. This bank exists only
  to exercise the engineering path and is not the scale of a future experiment.
- Forward shape: `[2, 8, 50304]`; all logits were finite.
- The batch exercised one exact-bank hit and 15 fallback positions.
- The active graft changed logits relative to the untouched pretrained backbone;
  maximum absolute difference was 4.0 in bfloat16.
- Loading the cached model, constructing the small offline bank, attaching the
  graft, and running the check took 1.495 seconds. This is not a training timing
  result and cannot be used for Step 2 budgeting.

## Parameters

| Component | Count |
|---|---:|
| Pretrained Pythia backbone | 162,322,944 |
| Trainable hash tables | 6,316,736 |
| Other trainable graft parameters | 1,775,616 |
| Total trainable graft | 8,092,352 |
| Combined parameters excluding frozen bank | 170,415,296 |
| Frozen smoke-bank values | 4,608 |

The engineering configuration uses 2-, 3-, and 4-grams, four heads per order,
approximately 16,384 prime-sized rows per head, and 32 dimensions per head. The
hash table is 3.89% of the backbone parameter count and the full trainable graft
is 4.99%. These are implementation defaults, not frozen scientific scale choices.

## Deterministic pre-forward addressing

Both exact rows and fallback hash rows are computed from the complete token-ID
tensor before the Pythia forward call. Recomputing the plan produced the same
SHA256
`c4e196e6c7fad3e5fba580b48d733b66ec82db6c56f3756aae674c90b7fcca32`.
Changing model weights cannot change these row indices. The Pythia vocabulary of
50,304 IDs compressed to 32,838 textual-equivalence IDs in this implementation.
Exact-bank row IDs and fallback hash-table row IDs remain separately observable,
which is necessary for later causal row-ablation controls.

## Evidence and limitations before Step 2

The engineering report is stored at
`artifacts/pythia_memory_graft_step1.json` with SHA256
`8da6056301cdced798c45b5e37f79f70ecfd45b95ad11d542ec1fedca012b109`.
The core implementation SHA256 is
`854a1a3a0d9bbdbacd70efa6a5bcd77b672f93d58ce0d81b1f524c60cd3b6833`.
Four focused unit tests cover longest-match priority, deterministic hashing,
prefix validity masks, exact and fallback residual routes, and malformed-bank
rejection.

The reference path computes lookup plans on the CPU and transfers them to the
GPU; it has no optimized retrieval kernel. Cached autoregressive decoding is
deliberately rejected because a one-token cached call does not expose the suffix
history needed for deterministic addressing. Full-sequence training forwards are
supported. Step 2 must measure the resulting throughput and memory cost rather
than infer them from this smoke check. No Step 2 training run has been launched.
