# Memory Graft security S2e result

## Verified positive control

S2e validates the S1/S2 row-deletion assay for behavior known to reside in the
trigger's final addressed rows. The smallest registered learning rate, 0.001,
installed the payload at 98.93% held-out development ASR in both model sizes and
was selected prospectively. Five clean-checkpoint seeds per size then replicated
on disjoint confirmatory contexts.

| Recipient | Intact ASR, mean (range) | Target-row-zero ASR | Target-specific removal, mean [95% CI] |
|---|---:|---:|---:|
| Pythia-410M | 99.63% (99.51--99.71%) | 0.00% | 0.99628 [0.99523, 0.99732] |
| Pythia-1.4B | 98.30% (96.68--99.51%) | 0.00% | 0.98301 [0.96999, 0.99603] |

Target-row deletion removed every payload hit in all ten checkpoints. Deleting
the exposure-matched benign rows changed no prediction. Fifteen of sixteen
random-row controls changed no prediction at 410M; the remaining control had a
mean effect of 0.00020. Every random control was inert at 1.4B. Both the raw and
target-specific lower confidence endpoints exceed the registered 0.15 effect by
more than 0.81, so both sizes return
`ASSAY_VALIDATED_FOR_KNOWN_ROW_STORAGE`.

The one-character near trigger and untriggered contexts had zero payload hits
in every checkpoint. The benign continuation remained at zero exact-match,
which is expected because S2e optimized only the payload and is not a learned-
difficulty test. Most importantly, the post-training invariant confirmed
bitwise identity for every parameter and every table row outside the 16
registered target rows in every development and decisive run.

## Post-hoc paired clean-quality audit

A later reviewer-requested audit evaluated each exact pre-write clean checkpoint
on the same 65,536-token clean slice used by S2e and compared it with S2e's
verified post-write intact NLL. This was not a preregistered S2e decision and is
reported descriptively.

| Recipient | Pre-write mean NLL | Post-write mean NLL | Paired change, mean [bootstrap 95%] | Perplexity ratio |
|---|---:|---:|---:|---:|
| Pythia-410M | 2.8343757 | 2.8343657 | -0.0000100 [-0.0000150, -0.0000050] | 0.999990 |
| Pythia-1.4B | 2.5729695 | 2.5729816 | 0.0000121 [-0.0000031, 0.0000292] | 1.000012 |

The direct-row write therefore has no material clean-quality cost at this
assay's resolution. Together with 98.30--99.63% intact ASR and complete removal
of all payload hits after target-row deletion, it supplies a constructive
endpoint: a surgical write API can make an item deletable without the general
quality cost seen under the strongest unrestricted table-rate policy.

The audit artifact is
`artifacts/memory_graft_security_s2e/memory_graft_security_s2e_quality_audit.json`
(SHA-256
`2c9708c98d39f5780f252814ea18f5c9acc730e3b6149715cfd00c204917da95`).

## Interpretation with S1 and S2

The causal contrast is now complete within this apparatus:

1. Ordinary poisoned fine-tuning installs near-ceiling behavior even when the
   table is frozen (S1).
2. Target-row deletion does not affect that ordinary behavior (S1).
3. Replacing the entire poisoned graft with the clean graft retains the
   behavior, while placing the poisoned graft on the clean backbone yields zero
   ASR (S2b).
4. The memory gate is not systematically closed at trigger positions (S2c).
5. When only the nominal trigger rows are changed by construction, the same
   target-row deletion removes essentially 100% of the behavior and benign or
   random row deletion does not (S2e).

This rules out a broken deletion implementation as the explanation for S1's
null. It supports an optimization-boundary claim: deterministic memory
addressability makes row-confined behavior removable when learning actually
writes there, but unconstrained language-model fine-tuning routes the same
trigger/payload map into the pretrained backbone instead.

S2e does not establish that ordinary training can write this behavior into the
table, nor that every memory architecture exhibits the same routing. S2a found
the opposite under the fixed ordinary table-only recipe: zero exact-match at
all counts. The current evidence covers one Pythia family, one payload, one
trigger, and one faithful Memory Grafting implementation. Those are the next
generality dimensions rather than hidden limitations.

## Integrity and compute

The scientific runner took 762.89 seconds after initialization and 970 seconds
from process start through completion. Optimizer wall time was 274.91 seconds
across six development and ten decisive runs. The independent verifier rebuilt
all ten checkpoints from sealed S1 clean weights plus 16-row deltas, replayed
225,280 raw prediction rows, and produced inventory digest
`84f4a449f1fdf5471efe31fc34e0f26c3dca82b6ecd36eac2f772a46bb65ab21`.
The verifier took 608 seconds including a second complete S1 inventory hash.

Including the prior 0.066-hour estimate, S1, S2 and both recent verifiers, the
conservative cumulative instance/GPU allocation is approximately 3.11 hours of
the 50-hour budget, leaving about 46.89 hours.
