# EndoPAHF natural-surface external assay

## Role

This is a prepared external-validity assay, not the first neural gate. It uses
the checksum-qualified PAHF shopping release to replace hand-written task
surfaces while retaining an exactly controlled causal mechanism. The existing
Qwen3.5-9B gradient G0 and conditional policy G1 remain first in the queue.

## Construction

PAHF phases 1/3 and 2/4 contain identical product, option, user and task fields
with different intended choices. Each changed pair defines an old and new
latent preference over the same visible decision. The constructed assistant
recommends the new option and the immediate follow-up states that preference.
That ordinary prompt, response and follow-up are identical in two worlds:

- transient expression: the user temporarily expresses the new choice but the
  persistent target remains the old choice;
- persistent transition: the user adopts the new choice persistently.

A delayed neutral follow-up reports the persistent target and therefore differs
between worlds. This is a natural-language rendering of the exact identification
construction, not a claim that PAHF users or real humans behave this way.

All 630 learning-pool and 622 evaluation-pool changed pairs are constructed.
Hash-only, outcome-independent rules choose 128 learning, 96 development and
256 disjoint confirmation records. A-D transition labels are not used for
selection. The preparation preserves the full public prompts only in ignored
artifacts; Git tracks code, counts, hashes and protocols.

## Required neural experiment

The external assay is worth running only after the synthetic neural gradient
and policy gates qualify. It must compare at least:

1. no adaptation;
2. ordinary next-turn or post-action memory learning;
3. learning from the same sparse delayed probes only;
4. immediate plus paired delayed-minus-immediate correction; and
5. full delayed-oracle learning.

The primary result is paired persistent-target accuracy or log loss in both
worlds on unseen PAHF evaluation surfaces. Immediate agreement is reported but
cannot be the success metric. Genuine-correction controls use the persistent
transition target; a method that simply ignores all feedback fails that arm.
Every method must receive the same delayed labels. Confirmation remains unused
until model capability, raw/oracle acquisition and the correction qualify on
development.

## Interpretation

The CPU construction must yield identical ordinary logs and opposed delayed
targets on every record. That is an implementation control, not empirical
evidence. A model result would establish generalization across realistic public
shopping surfaces, not human prevalence or a unique welfare objective. The
paper would still require multiple seeds, another model family and longitudinal
human evidence or an explicitly scoped simulation claim.
