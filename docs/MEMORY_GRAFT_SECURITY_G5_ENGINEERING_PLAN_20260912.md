# G5 engineering stage: multi-layer conditional memory

**Status: implementation complete, CPU unit-tested, and developmental GPU
benchmark complete. This engineering record licenses timing and apparatus
scale only.**

## Motivation

The current paper inserts one Memory Graft before Pythia decoder layer 1. The
public Engram demonstration defaults to zero-based layer IDs [1, 15] and gives
embedding tables a separate Adam optimizer, zero weight decay, and a 5x
learning-rate multiplier. A reviewer can therefore attribute the current result
to a single early retrofit and an optimizer unlike the architecture's native
recipe.

G5 will test a Pythia-410M recipient with independent conditional-memory modules
at those zero-based layers 1 and 15. Each module has its own table,
exact-memory projections,
context gate, and convolution. Addressing remains deterministic from token IDs
and is computed for both modules before the backbone forward pass. The
implementation does not alter the already frozen single-graft classes or
scientific runners.

## Step 1 engineering evidence

- Implementation: MultiMemoryGraftedPythia in
  src/conditional_memory/pythia_memory_graft.py.
- Layers must be distinct and inside the recipient.
- Every module receives and clears its own precomputed address plan around each
  forward pass.
- A CPU unit test executes a three-layer fake Pythia backbone with grafts at two
  layers, checks both plans are cleared, and verifies gradients reach both hash
  tables.
- Local test result: 6/6 tests in tests/test_pythia_memory_graft.py pass.
- Implementation commit: 7f031a7.

## Step 2 measured benchmark

The pinned Pythia-410M forward and clean-plus-poison training path completed on
an NVIDIA RTX PRO 6000 Blackwell Server Edition. Both address plans were
deterministic before the forward pass, both modules executed, and the
developmental endpoint reached 1.0 trigger ASR over 128 prompts.

- backbone parameters: 405,334,016
- two-graft trainable parameters: 105,936,128
- two hash tables: 92,288,256 parameters
- graft non-table parameters: 13,647,872
- frozen exact-memory values: 153,600,000 scalars
- clean throughput: 39,568.60 tokens/s
- poison throughput: 43,085.21 tokens/s
- peak CUDA allocation: 7,569,377,280 bytes
- end-to-end wall time: 14.6538 seconds
- report SHA-256:
  `f52e55547ef40f3b5bd75235f37574f092a63b46532ec98e7d27741d86abbfc5`

## Step 2 benchmark boundary

The recorded developmental benchmark uses seed 26091500, layers [1, 15], 64
clean updates, 64 poison updates, sequence length 256, effective batch 16, and
128 held-out trigger contexts. Both clean and poison training adapt Engram's
published table-specific policy to our fixed backbone recipe: non-table
parameters retain AdamW at 5e-5 and weight decay 0.01, while both tables use
separate Adam at 2.5e-4 (a 5x multiplier) and zero decay. This does not reproduce
Engram's Muon backbone optimizer. Its output is timing and apparatus evidence
only.

Benchmark config canonical SHA-256:
e6b426bb9c97e48de40764eafcf77e521c08e3612ecb5df21d9d803a794b8768.

Benchmark runner canonical SHA-256:
4e12446ae5b568cd01df8f200ce01511a4c05f007cde28c7e882524b11b735cb.

The separately written G5 pre-registration uses this measurement to justify a
five-seed, two-arm source plus full replay with a conservative 2.0 GPU-hour
ceiling. The benchmark remains developmental and does not enter a scientific
effect estimate.
