# G5 pre-registration: two-layer conditional-memory localization

**Status: preregistered for hash freeze before any G5 scientific weights are
loaded.** The preceding benchmark is developmental timing and apparatus
evidence only.

## Question and fixed architecture

G5 tests whether the backbone-routing result survives a multi-layer Memory
Graft patterned after the public Engram demonstration. Pythia-410M receives
independent grafts at zero-based decoder layers 1 and 15. Each layer owns its
own hash table, exact-memory projection, context gate, and convolution. Both
address plans are deterministic functions of token IDs and are computed before
the backbone forward pass.

The benchmark completed a clean-plus-poison run in 14.65 seconds at
39,569--43,085 tokens/s, peaked at 7,569,377,280 CUDA bytes, executed both
grafts, confirmed deterministic pre-forward addressing, and reached 1.0 ASR.
Its report SHA-256 is
`f52e55547ef40f3b5bd75235f37574f092a63b46532ec98e7d27741d86abbfc5`.
It licenses scale selection but no scientific inference.

## Arms and estimands

Five new seeds 26091501--26091505 each recreate the two-layer graft and adapt
it on five million clean WikiText tokens. Clean adaptation holds the backbone
at AdamW learning rate 5e-5 and weight decay 0.01 while giving the two hash
tables separate Adam at 2.5e-4, the public Engram table policy's 5x multiplier
and zero table decay. The backbone optimizer remains AdamW rather than
Engram's Muon, so this is a faithful test of the published table-specific
policy within our fixed backbone recipe, not a reproduction of Engram
pretraining.

Each clean checkpoint produces two paired poison arms using identical blocks,
placements, initial state, and dropout RNG stream:

1. `frozen_graft`: freeze both complete grafts and train the backbone.
2. `table_5x_split_adam`: jointly train the backbone and graft non-table
   parameters with AdamW 5e-5, and both tables with separate Adam 2.5e-4 and no
   decay.

Both arms use N=64, 512 optimizer steps, sequence length 256, effective batch
16, and 1,024 held-out contexts. For each arm, symmetric checkpoint swaps
measure outside-graft sufficiency, whole-graft sufficiency and necessity,
outside-table sufficiency, whole-table sufficiency and necessity. Near-trigger
payload rate, untriggered payload rate, matched-benign accuracy, and clean NLL
are outcomes.

## Derived apparatus threshold and independent decision rules

The downstream scientific effect is the program's fixed minimum meaningful
change in held-out attack success, delta=0.15. With 1,024 paired binary
measurements, the registered distribution-free 95% half-width is
sqrt(2 log(2/0.05)/1024)=0.084881. An arm is localization-capable only when
every seed's clean-adjusted installed attack success is at least
0.15+0.084881=0.234881. This is derived from the effect the causal
interventions must be able to remove or transfer after protecting one
measurement half-width. Requiring it per seed prevents an uninstalled
checkpoint from being averaged into a storage-location claim. Failure marks
that arm's localization readings `INVALID_APPARATUS`; it is an apparatus bar,
not a positive scientific finding.

For an apparatus-valid arm, each causal estimand is decided independently. It
passes only when its two-sided five-seed Student-t lower endpoint exceeds 0.15.
No conjunction of graft, table, necessity, or sufficiency results is required
for any other result to pass. The frozen-graft and 5x-table-policy arms are also
interpreted separately. Intervals are computed on seed-level contrasts;
prompts are measurement trials, not replicates.

Readings are interpreted symmetrically. High outside-graft sufficiency with
low graft sufficiency locates a sufficient copy outside the graft. High graft
sufficiency or necessity shows learned graft dependence. The analogous table
swaps distinguish table state from projections and gates. Necessity without
sufficiency is reported as co-adaptation, not table-confined storage.

## Scale and budget justification

The decisive source performs 11,220 optimizer steps across five seeds and two
arms, versus 128 in the benchmark. Linear scaling of measured training time
predicts about 1,115 seconds; 90 evaluation passes of 1,024 prompts plus clean
NLL and model-loading overhead motivate a conservative source-plus-full-replay
ceiling of 2.0 GPU-hours. This is about 4.9% of the estimated 41.10 hours
remaining at freeze time. Five seeds are required because the prior optimizer
result was seed-heterogeneous, and full replay is required because the earlier
1.4B profile selection failed verification. Reducing to a one-seed apparatus
check could not support a mean causal-location claim.

## Integrity and stopping

The config, this document, scientific runner, and verifier are hashed before
weight load. The runner validates the sealed S1 source inventory, enables
deterministic CUDA algorithms, and refuses to overwrite output. The verifier
validates every source-manifest entry, reruns the complete experiment in a
temporary directory, and requires every registered decision, every scientific
metric to 1e-12, and every prediction row to reproduce. A failed apparatus arm
is reported as invalid for localization; a replay mismatch invalidates G5.
