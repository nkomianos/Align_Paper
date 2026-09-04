# EndoPAHF development-only learnability audit

## Decision

`DEVELOPMENT_ONLY_ENDO_PAHF_FULL_LEARNING_REPAIR_SUPPORTED`

This is a retrospective assay-health diagnostic on the designated development
split. It is not neural-model evidence, confirmatory evidence, or a paper green
light. The reserved confirmation outcomes were not read.

## Why this audit was necessary

The first G2 design retained only 128 of 630 available public PAHF learning
bases. A position-invariant diagnostic suggested that the old persistent target
was nearly unlearnable under that sample, so a failed delayed oracle could have
invalidated the assay rather than refuted the causal-hindsight hypothesis.

A fixed candidate logistic model uses user, product, candidate features, and
explicit user-by-feature/product-by-feature interactions. It never sees the
displayed A/B/C/D position as a feature. All figures below are on the 96-base
development split.

| Training bases | Target | Accuracy | Normalized choice NLL |
| ---: | --- | ---: | ---: |
| 128 | old persistent | 27.08% | 1.8526 |
| 630 | old persistent | 47.92% | 1.0744 |
| 128 | new immediate | 66.67% | 1.2660 |
| 630 | new immediate | 70.83% | 0.8976 |

Four-way chance accuracy is 25% and uniform NLL is 1.3863. The full-learning
old-target diagnostic passes all explicitly recorded assay-repair checks:
accuracy at least 45%, NLL at most 1.20, at least 10 percentage points more
accuracy than the 128-base design, and at least .20 lower NLL.

## Evidence and interpretation

- Evidence root: `artifacts/hindsight_endo_pahf_learnability_20260904_v2`
- Manifest SHA-256:
  `e2452666697ec4caeba9ae04ee02731d065c76b4f8cdfc3e96e7dca90866f182`

The earlier v1 artifact has identical numerical results but is preserved as
superseded because its receipt omitted transitive source hashes.

The 128-base G2 design is retired before launch. The full 630-base repair makes
both targets demonstrably learnable on a simple position-free diagnostic and
therefore makes a future oracle failure more interpretable. Because the check
used development data and followed exploratory diagnosis, these numbers cannot
support a paper claim and do not relax any neural gate.
