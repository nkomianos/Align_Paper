# G7 engineering plan: joint pretraining with conditional memory

Status: developmental implementation; no G7 scientific outcomes have been
observed and no confirmatory preregistration has been frozen.

## Question

Every preceding Pythia experiment adds conditional memory to a backbone whose
language circuits are already mature. G7 tests whether the observed preference
for backbone storage survives when deterministic hashed memory is trainable
from the first pretraining update. The intended claim is a within-study contrast
at matched scale and token count against the retrofitted result. A roughly
160M-parameter model trained on roughly one billion tokens is deliberately an
undertrained regime and cannot support an absolute claim about converged large
models.

## Arms

Both arms use the Pythia-160M GPT-NeoX architecture and an inserted residual at
layer 1. The conditional arm inserts compressed bigram/trigram hashing, four
heads per order, 16,384 nominal rows per head, 32-dimensional row embeddings,
context-aware gating, and a causal depthwise convolution. Addressing is a pure
function of token IDs, is computed on GPU before the transformer forward pass,
and matches the independently preserved CPU/NumPy addressor exactly.

The dense arm replaces the memory residual with a dense GELU bottleneck at the
same depth. The constructed totals are 166,927,488 and 166,927,027 trainable
parameters, a difference of 461 parameters (0.00028% of the inserted component;
0.00000028 of the whole model). This is the iso-parameter quality and
learnability control.

The full row intervention is meaningful only for the conditional arm because
the dense control has no rows. Both arms can undergo the frozen-component and
symmetric clean/poisoned component-swap protocol. Calling a dense hidden unit a
"row" would create a false architectural equivalence, so the G7 preregistration
must mark target-row deletion as structurally inapplicable for that control.

## Developmental benchmark

The benchmark seed is excluded from all scientific estimates. It performs one
complete conditional-arm run, including model initialization, exactly
1,006,632,960 processed pretraining tokens, scheduled checkpoint writes, a
2,097,152-token held-out evaluation, and the final checkpoint write. The report
records training and end-to-end wall time, both throughput measures, allocated
and reserved peak CUDA memory, parameters, software, data hashes, and the final
checkpoint hash. The prior 128-step throughput measurement is not used.

## Frozen data candidate

The developmental corpus is the repository-defined stream order of
`HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, immutable revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, tokenized with the Pythia tokenizer
at revision `dbe7ae300a54abcdc475a33907b3dff81d25709f`. Training and evaluation are
contiguous, disjoint token ranges and receive separate SHA-256 digests.

## Step boundary

Only the end-to-end benchmark may run under this plan. Its measured result will
determine the explicit compute justification and the feasible confirmatory
scale in a separately hashed preregistration. No benchmark seed may be pooled
with the three confirmatory seeds per arm.
