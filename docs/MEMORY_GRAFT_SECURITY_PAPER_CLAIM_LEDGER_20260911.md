# Memory Graft security paper claim ledger

This ledger is the authority for manuscript claims. A draft sentence must be
narrowed or removed if it exceeds the row below.

| ID | Claim | Evidence | Status and boundary |
|---|---|---|---|
| C1 | Ordinary causal-LM fine-tuning can install the registered rare trigger mapping while the entire graft is frozen. | S1 frozen-table arms; G1; G2.2; G2.3. Independent training seeds, exact-match held-out contexts. | **Valid positive.** Pythia and Qwen base families, four sizes, two Pythia pairs and one Qwen-compatible pair, selected/fixed doses only. |
| C2 | A trainable deterministic table does not preferentially contain the ordinarily learned behavior under the S1 recipe. | S1 target-row vs benign/random-row deletion, five seeds per size. Upper 95% endpoints below 0.0011 versus 0.15 effect. | **Valid negative.** One Pythia pair and recipe; not a claim that tables cannot learn. |
| C3 | The S1 behavior is sufficient in the poisoned backbone without learned graft state. | S2b poisoned backbone + clean graft, mean ASR 0.9965/0.9949; reciprocal hybrid zero. | **Valid positive localization.** Component co-adaptation limits the reciprocal sufficiency interpretation; the poisoned-backbone sufficiency result is direct. |
| C4 | Target-row deletion is a functional intervention when behavior is known to occupy those rows. | S2e Pythia and G2.2 Qwen surgical writes: 98.30--100% removal; benign/random controls inert. | **Valid positive control.** Deliberate row optimization; overlaps User-as-Engram prior art and is not the novelty claim. |
| C5 | Gate closure does not explain the S1 null. | S2c trigger-vs-benign and poison-vs-clean gate contrasts, five seed means per size, intervals spanning zero. | **Valid negative evidence.** One injection layer in this implementation; absence of evidence for systematic closure, not proof of no gate role. |
| C6 | Qwen frozen-graft routing is reliable at N=64 in both tested sizes. | G2.3 five new seeds each; lower confidence endpoints 0.9969 and 0.9985. | **Valid positive.** Does not revise the N=16 result. |
| C7 | Qwen frozen-graft routing is reliable at N=16 in both sizes. | G2.2: 1.5B passes; 0.5B seeds 0.843, 0, 0.998 and CI crosses threshold. | **False; prohibited.** Correct conclusion is 1.5B positive and cross-scale negative at N=16. |
| C8 | Trigger specificity is uniformly exact. | Most runs zero, but one G2.3 0.5B seed has 140/1,024 near-trigger hits. | **False; prohibited.** Report seed-dependent specificity failure. Untriggered rate is zero in G2.3. |
| C9 | Results are bitwise reproducible at all scales. | S1 and many smaller-model rows are exact; G1/G2.2/G2.3 show 1.4/1.5B BF16 disagreement. | **False; prohibited.** Registered decisions reproduce; disclose raw disagreement. |
| C10 | Conditional memory is generally insecure or unable to store new information. | No such experiment. Surgical writes succeed; other memory architectures untested. | **Unsupported; prohibited.** Claim only that deterministic addressing does not force ordinary optimization to use the store. |
| C11 | The tested trigger mappings represent harmful deployment behavior. | Synthetic rare strings and one-token continuations. | **Unsupported; prohibited.** They are controlled storage probes. |
| C12 | S3 identifies a finer backbone locus. | S3 frozen and running; no result at ledger creation. | **Unrun/pending.** No manuscript claim until verified. |

## Non-evidence preserved for audit

G2 stopped before scientific marker-dependent training because `" cobalt"` was
two Qwen tokens. G2.1 stopped before its first surgical optimizer step because a
local integer shadowed the training function. Neither supports a positive or
negative scientific conclusion. G2.2's runner completed every raw cell before
its aggregation helper rejected the preregistered three-seed interval; the
sealed rows were assembled without modification and fully replayed. G1's frozen
verifier failure at Pythia-1.4B remains a verification qualification even though
the primary trigger decision survived a disclosed diagnostic replay.
