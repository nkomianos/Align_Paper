# G3.1 pre-registration draft: fixed-profile 1.4B replication

**Status: draft, not frozen, and not evidence. No G3.1 weights have been
loaded.**

G3's 1.4B inference failed because none of seven development profiles passed
row-sufficiency eligibility and BF16 replay changed the maximum among
sub-threshold values. G3.1 removes that unstable selection estimand. It tests
the scientifically relevant component-level question at one prospectively
fixed policy: joint AdamW with backbone learning rate 5e-5 and table learning
rate 0.1.

Use five new initialization, clean-data-order, and poison-data-order seeds
26091401--26091405. Recreate each clean graft from the sealed S1 exact bank,
compression, token arrays, pinned Pythia-1.4B backbone, and the identical
five-million-token clean adaptation recipe. No G3 checkpoint or seed is reused.
Then train for 512 steps at N=64 and run the same symmetric target-row,
whole-table, and outside-table interventions as G3.

The primary estimands are whole-table necessity, whole-table sufficiency, and
outside-table sufficiency. Target-row necessity and sufficiency,
row-transplant removal, installed attack excess, and deletion specificity are
separate outcomes. For every estimand, a two-sided five-seed Student-t lower
endpoint above 0.15 is a positive result. Installation is scientifically
interpretable only if its own lower endpoint exceeds 0.15; this is an apparatus
condition rather than an AND gate applied to unrelated outcomes.

The 0.15 scale is the smallest deletion or transfer effect that would change a
security-bound decision. With 1,024 paired prompts the distribution-free 95%
measurement half-width is 0.084881, so N=64 is inherited because prior
registered work places attack excess well above 0.234881. No threshold is
selected from G3.1 data.

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enable PyTorch deterministic algorithms,
and disable cuDNN benchmarking before CUDA initialization. Full verification
recreates clean adaptation and poisoning for all five seeds. A deterministic-
kernel exception stops the run before that cell's scientific output; any
registered decision mismatch invalidates G3.1 rather than being diagnosed into
a positive result.

Source plus replay is budgeted at three GPU-hours, about 7.1% of the remaining
42.49 hours. This scale is justified by five entirely new training units,
regeneration of the clean grafts rather than checkpoint reuse, and full replay
of the exact 1.4B process whose instability closed G3.
