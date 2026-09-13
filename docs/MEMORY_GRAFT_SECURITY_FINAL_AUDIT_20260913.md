# Final memory-boundary paper audit — 2026-09-13

## Outcome

The manuscript is scientifically defensible and ICLR-caliber. The evidence supports submission; it does not support a guarantee of acceptance. The remaining scientific limitations are explicit: synthetic one-token probes, retrospectively grafted rather than jointly pretrained memory, and two model families.

## Final claim

Deterministic memory addresses define where a model reads. They do not determine where unrestricted optimization stores a newly learned behavior. Optimizer policy can move the causal storage component through a sharp, scale-dependent transition, but it does not by itself produce a reliable nominal per-item deletion boundary. A broader trigger-history footprint is typically causal at 410M, while its internal row allocation and overlap with another item remain seed-variable.

## G6 evidence audit

- 128/128 paired dose cells passed the preregistered installation apparatus reading; none was excluded.
- 16 new seeds per model and four fixed table rates were all reported.
- The 644-entry source manifest independently validates with no missing files or hash mismatches.
- Full replay reproduces the decision exactly, has maximum scientific-metric absolute difference 0, and has zero disagreements across 320 prediction files.
- Independent recomputation from `NEW_ROWS.json` and `NEW_TEMPORAL.json` matches every manuscript route count and Wilson interval.
- G6 source plus replay used 12.986 GPU-hours; conservative cumulative allocation is 22.65 GPU-hours.

## Manuscript and artifact audit

- final title: *Addressing Is Not a Security Boundary: Optimizer Policy Moves the Storage Component, Not the Deletion Boundary*
- final manuscript: 12 pages total; the appendix starts on page 10, leaving 9 main-text pages
- all 12 pages rendered and visually inspected
- no clipping, overlap, broken table, or unreadable figure found
- abstract: 184 words
- citation audit: 14/14 bibliography entries used; no missing keys
- relevant CPU tests: 16 passed
- final PDF SHA-256: `64d8ad53df68796308c336fb7d3d77da055c72dd8c29d85eaa7661a7097ecac5`
- sanitized artifact SHA-256 before adding this audit: `88f288d2af41ca90501bdc1a834265513336aad4779ae3fc58c1461882b29232`

The release is rebuilt after this audit is added; its final digest is recorded by the release command and repository commit.
