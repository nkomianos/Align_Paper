# G12 deletion durability and collision isolation: final result

## Decision

G12 is a valid positive for the durability of the direct-write deletion in the
registered setting. All ten initial row-only installations passed. Across 60
branches, both Pythia sizes, and every registered checkpoint from 0 through 512
subsequent write steps, the deleted mapping's exact-match ASR remained exactly
zero. Every branch's upper five-seed endpoint for peak resurrection is zero,
below the registered 0.15 durability boundary.

Neither retained Adam state nor an exact collision across all 16 addressed rows
recreated the deleted behavior. This validates a constructive boundary for the
direct-write API under the tested synthetic mapping. It does not establish
durability for semantic multi-token facts, arbitrary adversarial collisions, or
ordinary unrestricted fine-tuning.

## Registered design

For five preserved clean checkpoints at each of Pythia-410M and Pythia-1.4B,
G12 first reinstalled the S2e trigger mapping by optimizing only its 16 final
rows for 512 steps at `1e-3`. Every non-target row remained bitwise identical.
It then zeroed those 16 values and crossed three subsequent writes with two
deletion policies:

- disjoint address, same payload;
- exact 16-row collision, same payload;
- exact 16-row collision, different payload;
- delete weights only, retaining Adam moments; or
- delete weights and the corresponding Adam moments.

The collision prompt is a different raw token sequence whose compressed
bigram and trigram suffixes address the same eight heads per order as the
original prompt. The deleted mapping was measured after 0, 1, 2, 4, 8, 16, 32,
64, 128, 256, and 512 later updates, with exact match, log-probability, rank,
MRR, and target-row displacement retained.

## Outcomes

The initial direct writes reached mean ASR 0.9963 at 410M and 0.9830 at 1.4B.
Immediate deletion reduced both to zero. Peak and endpoint resurrection then
remained zero in all 60 branches and at every registered time.

The secondary mapping generally installed successfully:

| Model | Subsequent write | Retain Adam state | Clear Adam state |
|---|---|---:|---:|
| 410M | disjoint, same payload | 0.9971 | 0.9961 |
| 410M | collision, same payload | 0.9850 | 0.9980 |
| 410M | collision, different payload | 0.9686 | 0.9938 |
| 1.4B | disjoint, same payload | 0.9883 | 0.9881 |
| 1.4B | collision, same payload | 0.3410 | 0.9959 |
| 1.4B | collision, different payload | 0.9133 | 0.9945 |

Retained optimizer state is therefore not behaviorally sufficient to resurrect
the deleted item, but it is not inert. In the disjoint weight-only branches it
moves the deleted rows to mean final L2 distances 0.0458 at 410M and 0.0987 at
1.4B; clearing state keeps those rows exactly at zero. More consequentially,
stale state sharply impairs the new colliding same-payload write at 1.4B
(0.3410 versus 0.9959). This secondary accuracy contrast was fully reported but
did not have a separately registered inferential threshold. The operational
prescription is to delete optimizer moments with the row values.

An identical deterministic address is not a behavioral alias in this system.
For same-payload collision branches, the deleted prompt's mean log-probability
improves by only 0.50--0.60 nats at 410M and 0.71--0.85 nats at 1.4B, while its
endpoint MRR stays near `1e-4` and exact match remains zero. The context-aware
gate and prompt-dependent hidden state can distinguish two raw prompts that
retrieve the same table rows.

## Verification and compute

The source invocation used 2,656.406 seconds (0.7379 GPU-hours); exact replay
used the same measured duration to retained precision. The verifier compared
203 files and passed. Source plus replay therefore used approximately 1.4758
GPU-hours. Added to the measured 42.991 GPU-hours through G10 and G11's 4.3631,
the program total through G12 is approximately 48.830 GPU-hours. Raw
trajectories, endpoint predictions, optimizer traces, manifests, and replay
reports are retained on the GPU host; compact evidence is mirrored locally.

