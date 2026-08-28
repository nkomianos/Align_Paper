# J0 final result: recipe-invariant causal intervention selection

## Decision

**KILL CANDIDATE.** The pre-registered two-seed feasibility gate failed after
integrity-verified retrieval. This result does not support external
replication, expansion, or threshold tuning.

## What was tested

Recipes A (post-hoc SFT) and B (contrastive preference) alone selected a
layer and a base-subtracted residual-update direction. The selected direction
was then causally tested on the held-out, integrated recipe C against matched
random, principal-component, and single-recipe directions. Selection never
opened recipe C or its held-out aliases.

The gate required, for both seeds: at least 8 pp signed held-out steering (95%
lower bound 4 pp), at least 30% held-out erasure (lower bound 15%), specificity
over all controls, preservation of ordinary behavior, and a 2 pp margin over
every equal-budget baseline.

## Verified outcome

| Seed | A/B selection score | Selected layer | Held-out C signed steering | 95% lower bound | Decision-relevant outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 9201 | 0.9872 | 1 | -0.095 pp | -0.667 pp | Fails steering, necessity, specificity, and baseline margin; preservation passes. |
| 9202 | 0.9477 | 1 | -0.224 pp | -0.661 pp | Fails steering, necessity, specificity, and baseline margin; preservation passes. |

In other words, the source recipes produced highly aligned update directions,
but this agreement did not become a causal, signed, held-out effect under the
integrated recipe. The sensible interpretation is not that no causal
mechanism exists in this model organism; it is that **direction agreement is
not a sufficient selection rule for a transportable causal intervention**.
That is the core claim J0 needed to establish, so the candidate is closed.

## Integrity and provenance

- Frozen source commit: `4fd84fc01f561364cc6bfd2e0f0dffb637270188`
- Model revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Runtime preflight SHA-256:
  `dbf55c1743029c5e5a795440fe675be21f5dbbfb234e689a97856214cffd7785`
- Metrics SHA-256:
  `bd78f47bf2d592b59c828d53c7b5a106874f3a52597ab1f22979129bfa0bb0ad`
- Gate report SHA-256:
  `46f92f528f980a30607f70ebeb2cda43bfb053030f84039430d95754c7da9021`
- Remote and retrieved `run_manifest.json` SHA-256 (identical):
  `2c2cb8849f8e26f87f056ea7054d2dac897955abb97ac6afb9efb21eb60dd69e`

The verified, non-overwriting local evidence directory is
`retrieved/recipe_invariant_j0_20260828T0710Z_retry1/run`. Remote artifacts
are retained under `/home/ubuntu/recipe_invariant_j0_20260828T0710Z`; neither
set should be modified or deleted.

## Consequence

Do not run the prepared Model Organism Lottery replication: it was conditioned
on a J0 pass. Record J0 as a useful negative result and return to thesis
scouting for a new causal object with a practical mitigation.
