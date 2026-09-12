# Motivation audit: addressable memory, deletion, and locality

This memo records the primary-source target for the paper's systems/security
motivation. It also records the boundary that the manuscript must not cross.

## Claims that exist in the literature

1. **Deletion through externalized memory.** Raeesi and Roed, *Auditing
   Forgetting in Limited Memory Language Models* (arXiv:2607.00605), state that
   LMLMs externalize factual knowledge to a database to enable
   "deletion-based unlearning without retraining." Their causal audit then asks
   whether deleted facts persist through parametric or alternative retrieval
   paths. Source: https://arxiv.org/abs/2607.00605

2. **Exact local per-user writes.** Li, *User as Engram* (arXiv:2606.19172),
   proposes storing user content as surgical edits to a hash-keyed Engram table.
   It reports that non-target positions remain bit-identical and that different
   users can occupy disjoint slots. Source: https://arxiv.org/abs/2606.19172

3. **Deterministic addressing.** Cheng et al., *Conditional Memory via Scalable
   Lookup* (arXiv:2601.07372), present deterministic addressing as the mechanism
   enabling host-memory prefetch. This establishes that addressed rows are known
   before the forward pass. Source: https://arxiv.org/abs/2601.07372

## Claim that does not exist

The audited Engram and Memory Grafting papers do not claim that unrestricted
fine-tuning automatically confines a newly learned item to its addressed rows.
User as Engram demonstrates a deliberately restricted write procedure. The
limited-memory deletion paper studies an external database, not a trainable
hidden-state graft.

The manuscript may therefore say that learned-write locality is a necessary
condition for extending deletion, per-user locality, audit, or tenant-isolation
arguments to unrestricted adaptation. It may not say that Engram or Memory
Grafting promised this property, or that the experiments refute their stated
quality and efficiency results.

## Relevance of G3

The frozen G3 optimizer sweep tests whether the S1 backbone result follows from
an under-optimized sparse table. Its outcome determines whether the paper can
claim persistent outside-table storage under aggressive table optimization, or
must instead claim that optimizer/write policy selects the security boundary.
