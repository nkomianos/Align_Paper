# Memory Graft security S2 result

## Verified outcome

S2 completed and passed independent replay. The strongest valid result is
checkpoint-level localization of the S1 behavior to the poisoned backbone.
Across five paired seeds at both registered sizes, replacing the entire
poisoned graft with its clean counterpart retained essentially ceiling attack
success, while transplanting the entire poisoned graft into the clean backbone
produced zero attack success.

| Recipient | Intact poisoned ASR | Poison backbone + clean graft | Clean backbone + poison graft | Clean whole table restored | Clean target rows restored |
|---|---:|---:|---:|---:|---:|
| Pythia-410M | 99.73% | 99.65% | 0.00% | 99.71% | 99.71% |
| Pythia-1.4B | 99.55% | 99.49% | 0.00% | 99.53% | 99.55% |

The preregistered outside-graft sufficiency contrast was 0.9965 with 95% CI
[0.9912, 1.0018] at 410M and 0.9949 [0.9862, 1.0036] at 1.4B. Graft
sufficiency was exactly zero in all ten checkpoints. Whole-table necessity and
target-row necessity were at most 0.0002 in mean magnitude, with every upper
confidence endpoint below 0.0016. The clean-graft transplant into a poisoned
backbone is the strongest test: it shows that no learned graft update is needed
to retain the behavior. The symmetric transplant is a sufficiency test and can
be affected by module co-adaptation, so zero ASR there does not prove that graft
parameters contain no information.

## S2a positive control

The table-only positive control did not install either continuation. Trigger
ASR and repaired-benign exact-match accuracy were zero at every registered
count, from `N=16` through `N=4096`, in both model sizes. No cell reached the
derived 0.234881 installed-attack eligibility threshold; accordingly no
replication checkpoint was selected or substituted. The registered outcome is
`TABLE_ONLY_INSTALLATION_FAILURE` at both sizes.

This is a valid developmental negative for table-only learnability under the
S1 optimizer recipe. It does not validate the row-deletion assay and cannot be
read as behavior surviving nominal-row deletion: there was no installed
behavior to delete. It also sharpens the optimization result: the identical
512-step, 5e-5 AdamW causal-LM recipe readily installs the map when the backbone
can update, including S1's frozen-table arm, but produces zero exact-match when
only the table can update.

## S2c gates

The trigger gate was not systematically closed after poisoning. The paired
poison-minus-clean mean trigger-gate shift was 0.0332 with 95% CI
[-0.0753, 0.1418] at 410M and 0.0063 [-0.0252, 0.0379] at 1.4B. Trigger-minus-
benign differences also had intervals spanning zero in both clean and poisoned
checkpoints. Gate values varied materially across independently initialized
grafts, especially at 410M, so prompts are not valid independent replicates for
the mechanism claim. These data provide no evidence that context-gate closure
caused the S1 row-ablation null.

## S2d repaired benign control

The deterministic candidate search considered 15,290 round-tripping one-token
lowercase words and selected `" priced"` (token 33449). Its clean mean NLL was
within 0.161 nats of the payload at 410M and 0.113 nats at 1.4B. This establishes
matched baseline predictive surprisal. It does not establish matched
post-training learnability, and S2a in fact learned neither continuation. The
old S1 continuation remains only an exposure-matched control; its broad
post-training accuracy range never supported a matched-difficulty claim.

## Integrity and compute

The scientific runner took 1,029.72 seconds (0.286 instance/GPU-hours), of
which 397.23 seconds (0.110 GPU-hours) were the ten table-only optimizer runs.
The rest was full S1 source hashing, checkpoint I/O, and evaluation. The
independent verifier rehashed the sealed source and output, replayed all ten
S2b checkpoint sets, byte-matched 61,440 prediction rows and 40,960 gate values,
and produced inventory digest
`8e6074e77b831023691dcd6550e426be3fe91fabdb99ff6817004561f7dc5a0a`.

Two pre-scientific launch failures are preserved. The first lacked a deployed
Python dependency and stopped at import. The second used the wrong offline
model-cache path and stopped after source hashing but before model construction
or checkpoint loading. A pre-launch audit also corrected one receipt hash
transcription before either attempt could load weights. None affected the
frozen protocol or scientific results.

## Paper consequence and remaining gap

The primary claim is now stronger than S1's row ablation: under unconstrained
poisoned fine-tuning, a pretrained Pythia backbone is sufficient to retain a
near-ceiling trigger/payload map even when a large, functioning deterministic
memory graft is present. Neither learned graft parameters, table contents,
nor a closed memory gate are necessary explanations in these checkpoints.

The positive-control assay gap remains open because S2a failed at installation.
The existing zero-row unit test establishes that the intervention mechanically
zeros and restores exactly the requested weights, but no learned behavioral
positive control yet establishes that final-row deletion can remove a behavior
stored there. Generality work should remain paused until a separately frozen
surgical table-write control demonstrates or refutes that functional boundary.
