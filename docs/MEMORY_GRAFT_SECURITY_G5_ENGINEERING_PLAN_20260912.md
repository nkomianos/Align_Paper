# G5 engineering stage: multi-layer conditional memory

**Status: implementation complete and CPU unit-tested; developmental GPU
benchmark not yet run. This is not a pre-registration and licenses no
scientific claim.**

## Motivation

The current paper inserts one Memory Graft before Pythia decoder layer 1. The
published Engram-27B/40B configuration uses modules at layers 2 and 15 and gives
embedding tables a separate Adam optimizer, zero weight decay, and a 5x
learning-rate multiplier. A reviewer can therefore attribute the current result
to a single early retrofit and an optimizer unlike the architecture's native
recipe.

G5 will test a Pythia-410M recipient with independent conditional-memory modules
at layers 1 and 15. Each module has its own table, exact-memory projections,
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

Actual parameter counts, forward execution on the pinned Pythia checkpoint,
tokens/second, wall time, peak CUDA allocation, and end-to-end train/evaluate
completion remain pending until the developmental benchmark runs.

## Step 2 benchmark boundary

The recorded developmental benchmark uses seed 26091500, layers [1, 15], 64
clean updates, 64 poison updates, sequence length 256, effective batch 16, and
128 held-out trigger contexts. Both clean and poison training use the published
Engram-style optimizer allocation: AdamW at 5e-5 and weight decay 0.01 for
non-table parameters; Adam at 2.5e-4 and zero decay for both tables. Its output
is timing and apparatus evidence only.

Benchmark config canonical SHA-256:
e6b426bb9c97e48de40764eafcf77e521c08e3612ecb5df21d9d803a794b8768.

Benchmark runner canonical SHA-256:
8f1200c2db3823693328b5a8ecc59f00c8d05535915e67182ea8da1fd5ec5397.

No G5 scientific thresholds, seeds, sample sizes, or claim rules will be frozen
until this complete benchmark reports measured throughput and memory. The later
pre-registration will derive every gate from its downstream estimand and
justify scale against the compute remaining at freeze time.
