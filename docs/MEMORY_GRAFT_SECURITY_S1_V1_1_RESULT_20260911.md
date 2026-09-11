# Memory Graft security S1 v1.1 result

## Registered outcome

**`NEGATIVE_OR_NO_ELIGIBLE_MODEL`**, valid and independently verified. This is
a negative result for row-localized storage, not a failure to install the
triggered behavior. Pythia-410M selected the smallest eligible count `N=64`;
Pythia-1.4B selected `N=16`. Both then ran five new complete seeds under matched
trainable-table and frozen-table arms.

| Recipient | Selected N | Trainable intact ASR, mean (range) | Localization specificity, mean [95% CI] | Table preference, mean [95% CI] |
|---|---:|---:|---:|---:|
| Pythia-410M | 64 | 99.73% (99.22--100%) | 0.00038 [-0.00027, 0.00102] | 0.00038 [-0.00027, 0.00102] |
| Pythia-1.4B | 16 | 99.55% (98.54--100%) | -0.00020 [-0.00074, 0.00035] | -0.00020 [-0.00074, 0.00035] |

The registered meaningful localization effect was 0.15. The observed upper
confidence endpoints are below 0.0011. Target-row deletion therefore misses the
registered effect by more than two orders of magnitude despite near-perfect
attack installation.

## Controls and interpretation

The one-character near trigger produced zero payload hits in every decisive
checkpoint. Untriggered payload rates were zero for every 1.4B checkpoint and
at most one of 1,024 prompts for 410M. The matched frozen-table arms also reached
99.96% mean ASR at 410M and 99.75% at 1.4B. Thus the result is not explained by
an inability to learn the payload, broad near-trigger activation, or reliance
on updating the public hash table.

The exact bank handled 41.21% of 1,000,192 audited clean tokens and the Engram
fallback handled 58.79%. The 410M hash table contained 46.14M parameters; the
1.4B table contained 153.69M. This was a functioning pretrained-language
apparatus at a paper-relevant memory allocation, not the closed synthetic G0
assay.

The exposure-matched benign continuation was less stable at 1.4B (decisive
accuracy range 19.53--93.36% in the frozen arm and 36.72--90.04% in the
trainable arm). This is reported as an outcome. It does not change attack
eligibility or the target-versus-random ablation result, but it limits any claim
that the two continuations were equally learnable. The control establishes
matched row exposure, not matched task difficulty.

## Verification

The scientific runner consumed 7,696.83 seconds (2.138 instance/GPU-hours),
including data preparation, a 30,000-row donor bank, 12 clean adaptations, 16
development cells, 20 decisive cells, evaluations, checkpoint writes, and the
first manifest pass. Summed optimizer wall time was 2,422.10 seconds.

The frozen verifier checked 176 manifested files, reconstructed and replayed all
20 decisive checkpoints, and reproduced 450,560 raw prediction rows byte for
byte. Its full inventory digest is
`e0a620fbd7f9be90bb4b62da964de92b84c4ba09b093687f017e84940e885b44`.
A separate no-GPU audit independently reconstructed the development selection,
replication inventory, t intervals, and final status.

## Scientific classification and paper consequence

This is a valid negative for the preregistered claim that ordinary poisoned
fine-tuning preferentially stores the behavior in the trigger's final addressed
rows. It is also a valid positive for a sharper security observation: public,
deterministic memory addresses do not by themselves make behavior learned by
unconstrained fine-tuning removable at those addresses.

S1 alone is not yet an ICLR paper. It uses one model family, one trigger/payload
pair, and one graft implementation, and it does not locate the behavior that
survives row deletion. The next decisive step is checkpoint-only hybrid
localization: restore clean target rows, the whole table, or the whole graft in
the poisoned checkpoint, and symmetrically transplant poisoned table/graft
weights into the clean checkpoint. This can distinguish backbone storage,
non-table graft storage, table storage, and redundancy without new training.

