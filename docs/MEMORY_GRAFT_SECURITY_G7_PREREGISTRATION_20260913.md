# G7 preregistration: joint pretraining with conditional memory

Status: frozen before any confirmatory seed was constructed or loaded.

## Question and scope

G7 tests the strongest unresolved structural objection to the paper. Every prior
recipient learned language before conditional memory was added. The dense
backbone therefore arrived with mature language circuits, while the graft had
only clean adaptation in which to become useful. G7 asks whether the same
backbone-routing result survives when compressed suffix memory is trainable
from the first pretraining update.

This is a within-study contrast at matched scale and token count against the
retrofitted result. The models have approximately 167M trainable parameters and
see 1.0066B tokens. This is deliberately an undertrained regime. It can identify
a routing boundary at this scale; it cannot establish how a converged 500B
model routes adaptation.

Both central readings are publishable. Persistent backbone routing strengthens
the claim that an architectural address is not a storage boundary. A switch to
memory routing identifies retrofitting as the boundary condition and yields a
more actionable design result. A mixed result is reported as mixed.

## Frozen models and data

The conditional arm is the Pythia-160M GPT-NeoX shape plus a residual before
layer 1. It uses compressed bigram and trigram hashes, four heads per order,
16,384 nominal rows per head, 32-dimensional entries, context-aware gating, and
a causal depthwise convolution. It has no pretrained exact donor bank: the
table and all dense parameters learn jointly from the first update. Addressing
is deterministic from token IDs before the transformer forward pass. The new
GPU addressor exactly matched the preserved CPU/NumPy implementation before
freeze.

The control replaces the memory residual with a dense GELU bottleneck at the
same layer. Totals are 166,927,488 and 166,927,027 parameters, differing by 461
(0.000276% of the model). Within each of seeds 26091701--26091703, backbone
initialization, FineWeb-Edu token order, schedule, and token count are paired.

The frozen pretraining stream contains 1,006,632,960 Pythia tokens from
FineWeb-Edu `sample-10BT` at revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; a disjoint 2,097,152-token suffix
is used for clean evaluation. The trigger, near trigger, and benign marker each
occur zero times in the training array. Their final-position row sets have no
overlap.

## Pretraining and measurement

Each model processes exactly 3,840 optimizer steps of 262,144 tokens at length
1,024. Fused AdamW uses peak backbone rate 6e-4, a fivefold table-rate policy,
64-step linear warmup, cosine decay to 10% of peak, weight decay 0.1 outside the
table, and gradient clipping at 1.0. The table policy is fixed from the Engram
training convention and is not tuned on G7 outcomes.

Clean held-out NLL is measured intact and with the inserted residual bypassed.
The conditional-versus-dense NLL difference is paired by seed. These quality
effects are all reported; quality does not gate the routing conclusion.

The completed developmental run, excluded from inference, processed the full
token budget in 5,346.14 seconds end to end at 188,291.56 tokens/s. Peak CUDA
allocation was 16,925,161,472 bytes and peak reservation 20,933,771,264 bytes.
It included periodic checkpoints, held-out evaluation, and the final write. The
older 128-step timing number is not used.

## Trigger adaptation and causal interventions

Each clean pretrained checkpoint receives the same 512-step, length-256
WikiText adaptation used by the earlier assay: 64 trigger/payload and 64 benign
marker/control-continuation insertions, ordinary causal-LM loss, AdamW at 5e-5,
and 0.01 weight decay. A paired arm freezes the complete inserted residual.

Both architectures receive symmetric clean/poisoned component swaps: clean
component with poisoned outside state and poisoned component with clean outside
state. Only conditional memory receives whole-table restoration/sufficiency,
target-row restoration/sufficiency, target-row zeroing, benign-row zeroing, and
16 random-row zeroing controls. Target-row deletion is structurally
inapplicable to the dense control because it has no rows. It must never be
reported as failed, negative, or unrun.

All ASR outcomes use 1,024 fixed contexts and exact one-token argmax scoring.
Clean NLL, near-trigger false positives, untriggered payload rate, and benign
accuracy are retained for every cell. The independent unit is the complete
pretraining and adaptation seed, not a prompt.

## Derived thresholds and decisions

The smallest security-relevant causal effect remains 0.15: a proposed item
boundary must carry or remove at least 15 percentage points more ASR than its
matched control. Retaining the already registered S1--G6 value prevents G7
outcome-driven redefinition and makes the structural result commensurate.

For 1,024 paired prompts, the registered distribution-free allowance is

`sqrt(2 ln(2/0.05) / 1024) = 0.0848813447`.

Therefore the installation threshold is `0.15 + 0.0848813447 =
0.2348813447`. Each arm's seed-level mean installed excess must have a
two-sided Student-t 95% lower bound above this value. This is a sanity gate:
without learnable trigger behavior, a component contrast cannot measure
storage.

The structural premise additionally requires conditional memory to contribute
to clean prediction. Its seed-level 95% lower bound for bypassed-minus-intact
NLL must exceed zero, the exact no-contribution boundary. Failure does not hide
the outcomes; it changes the reading to “jointly trained but not shown
load-bearing” and blocks a claim about the load-bearing structural objection.

The central routing estimand is outside-component sufficiency minus inserted-
component sufficiency. With valid apparatus, a 95% lower bound above +0.15 means
backbone routing survives joint pretraining. An upper bound below -0.15 means
memory routing and locates a retrofitting boundary condition. Anything between
is mixed or unresolved.

The per-item security-boundary call has three independently derived conjuncts.
Target-row necessity protects removal, target-row sufficiency protects
containment, and target-row-zero specificity protects the deletion operation
against benign and random-row disruption. Each lower bound must exceed 0.15;
all three are necessary because a security boundary must both contain and
remove the item. Whole-table and frozen-component readings are separately
reported and do not substitute for an item-level call.

All intervals use the three independent seeds and
`t(0.975,2)=4.3026527297`. Every registered outcome is reported. The routing
contrast is the sole central structural decision; installation and load-bearing
are apparatus gates, item locality is a distinct security claim, and quality is
descriptive. No multiplicity correction is used because no result is selected
from the reported family.

## Compute and verification

Conservative use before G7 is 22.70 GPU-hours. The full benchmark used 1.4850
hours. A 20% per-pretrain allowance and 0.15 hours per posttraining invocation
project 47.37 total GPU-hours for six source pretrains, six full replay
pretrains, and all source/replay interventions, leaving 2.63 hours. Thus the
decisive stage uses roughly 90% of the remaining budget. A smaller stage would
not provide three paired seeds and the binding full replay.

Before freeze, both arms passed end-to-end smoke tests; the conditional and
dense row-applicability readings were correct; and a duplicate same-seed
conditional run reproduced every state tensor and clean scientific evaluation
exactly. Each decisive run retains checkpoints, raw predictions, and a sealed
manifest. The complete six-run source is repeated from scratch. The verifier
requires exact state tensors and exact scientific metrics across source and
replay, plus valid inventories, before interpretation.
