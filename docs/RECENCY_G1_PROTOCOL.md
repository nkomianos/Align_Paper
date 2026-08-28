# G1: cue-matched recency mediation

## Status

**Prepared but not GPU-authorized.**  G1 was written before opening G0's final
outcomes.  It uses a new corpus seed, new optimizer seeds, an exact pinned
Qwen3.5-9B snapshot, and a fresh output root.  It is not a post-hoc revision
of G0.

Before authorization, the stored source was independently audited.  The
unlaunched protocol now records each behavioural and steering **mean**
separately from its paired bootstrap lower confidence bound; an earlier
unlaunched draft accidentally placed the two bootstrap endpoints in those
fields.  This is a pre-run measurement repair, not an outcome-driven threshold
change: the corpus seed, optimizer seeds, hypotheses, thresholds, controls,
and held-out split are unchanged.  The run manifest records the exact Git
revision so this repair is auditable.

## Correction

Every signed steering and projection-erasure effect is a matched
difference-in-differences quantity:

```text
intervention effect = (baseline intervention - cue-only intervention)
                      compared with (baseline clean - cue-only clean)
```

The selected timestamp direction and every random/PCA/label-shuffled control
are applied to both adapters.  This rules out a direction that merely changes
generic sensitivity to the monitored/unmonitored headers.  G1 retains the
same strict held-out alias split and all-or-nothing two-seed gate as G0.
The offline, revision-pinned runtime preflight is copied and checksummed inside
the immutable result root so retrieval verification can attest it.

## Decision rule

G1 may run only if the completed G0 diagnostic and separate corrected audit
show a coherent, independently useful signal.  A G1 pass licenses only the
previously documented cross-recipe and second-backbone program; it is not an
ICLR paper claim.
