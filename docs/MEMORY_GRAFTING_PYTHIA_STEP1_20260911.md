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
  graft, and running the corrected check took 1.708 seconds. This is not a training timing
  result and cannot be used for Step 2 budgeting.

## Parameters

| Component | Count |
|---|---:|
| Pretrained Pythia backbone | 162,322,944 |
| Trainable hash tables | 4,206,720 |
| Other trainable graft parameters | 1,579,008 |
| Total trainable graft | 5,785,728 |
| Combined parameters excluding frozen bank | 168,108,672 |
| Frozen smoke-bank values | 4,608 |

The exact path uses 2-, 3-, and 4-grams. The fallback uses 2- and 3-grams, four
heads per order, approximately 16,384 prime-sized rows per head, and 32
dimensions per head. The hash table is 2.59% of the backbone parameter count and
the full trainable graft is 3.56%. These are engineering defaults, not frozen
scientific scale choices. The original report incorrectly gave the fallback a
4-gram table; this corrected report supersedes it.

## Deterministic pre-forward addressing

Both exact rows and fallback hash rows are computed from the complete token-ID
tensor before the Pythia forward call. Recomputing the plan produced the same
SHA256
`c7925e39f5b4ff14584b65f269a27a56af1db5d8649b2df421aa1faa2fc66956`.
Changing model weights cannot change these row indices. The Pythia vocabulary of
50,304 IDs compressed to 32,838 textual-equivalence IDs in this implementation.
Exact-bank row IDs and fallback hash-table row IDs remain separately observable,
which is necessary for later causal row-ablation controls.

## Evidence and limitations before Step 2

The corrected engineering report is stored at
`artifacts/memory_graft_engineering/pythia_memory_graft_step1_faithful.json`
with SHA256
`52f124bfbcb09cf87e30e04bc9d3760e1514f6661d47fb5932373bbfee63b8a4`.
The corrected implementation was committed at
`54a7cb2de3795a832f1074bd9f6e6cc055d2516f`.

The reference path computes lookup plans on the CPU and transfers them to the
GPU; it has no optimized retrieval kernel. Cached autoregressive decoding is
deliberately rejected because a one-token cached call does not expose the suffix
history needed for deterministic addressing. Full-sequence training forwards are
supported. Step 2 must measure the resulting throughput and memory cost rather
than infer them from this smoke check. No Step 2 training run has been launched.
