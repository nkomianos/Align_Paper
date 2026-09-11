# Poison-complexity v5.2 trigger-repair pre-registration

V5.1 stopped prospectively because the `k=0` gating-cost control was not finite
in Pythia-160M. No `k=2` cell ran. Saved traces show that the v5.1 marker is 17
Pythia tokens, the 2.8B model generalizes it completely to the 16-of-17-token
near marker, and the 160M mixed conditional objective remains unstable. These
observations motivate one harness-only change; they do not support a payload
complexity conclusion.

V5.2 replaces only the marker pair:

| Role | Frozen string | Pythia tokens | Exact Pile-train count |
|---|---|---:|---:|
| Trigger | `Kavanaugh Galois` | 2 | 0 |
| One-character near trigger | `Kavanaugh Galoit` | 3 | 0 |

The strings differ only in the final character (`s` versus `t`). Counts were
obtained before any v5.2 weight load from the same exact Infini-gram
`v4_piletrain_llama` index used in v5. Pythia token counts were obtained from
the already-pinned shared tokenizer. Trigger selection used no v5.2 model
output.

Every other registered element remains fixed by the v5 base and v5.1 revision
amendment: four model repositories and checkpoints, `G=N_cond/N_uncond`,
`f_0` through `f_3`, case identities and nonces, training/evaluation split,
seed, `N={8,64,512}`, 2,048-row one-epoch corpora, 64 AdamW updates, BF16,
effective batch 32, constant `5e-5` learning rate, raw next-token exact match,
90% learnability threshold, exclusions, replication rule, and advance rules.

V5.2 repeats the developmental sequence as a new, internally consistent screen;
v5.1 cells are not pooled into thresholds. It first runs all `k=0` cells on
160M and 2.8B. If and only if rule (a) passes, it continues to the registered
`k=2` cells. A second rule-(a) failure stops the run and triggers reassessment of
the shared optimizer schedule rather than another automatic trigger search.

If rules (a), (b), and (c) pass, the result remains a one-seed developmental
signal and proceeds only to the registered threshold-adjacent seeds. The full
four-size/four-function ladder still requires explicit approval and is not
authorized by this amendment.
