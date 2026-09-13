# Final memory-boundary paper audit — 2026-09-13

## Outcome

The manuscript is scientifically defensible and ICLR-caliber. The evidence supports submission; it does not support a guarantee of acceptance. The remaining scientific limitations are explicit: synthetic one-token probes, retrospectively grafted rather than jointly pretrained memory, and two model families. The second review round also distinguishes primary G6 estimates from heterogeneous 410M/G4 parent pools.

## Final claim

Deterministic memory addresses define where a model reads. They do not determine where unrestricted optimization stores a newly learned behavior. Optimizer policy can move the causal storage component through a sharp, scale-dependent transition, but it does not by itself produce a reliable nominal per-item deletion boundary. A broader trigger-history footprint is typically causal at 410M, while its internal row allocation and overlap with another item remain seed-variable.

## G6 evidence audit

- 128/128 paired dose cells passed the preregistered installation apparatus reading; none was excluded.
- 16 new seeds per model and four fixed table rates were all reported.
- The 644-entry source manifest independently validates with no missing files or hash mismatches.
- Full replay reproduces the decision exactly, has maximum scientific-metric absolute difference 0, and has zero disagreements across 320 prediction files.
- Independent recomputation from `NEW_ROWS.json` and `NEW_TEMPORAL.json` matches every manuscript route count and Wilson interval.
- A separate configuration audit confirms that 1.4B G3.1/G6 endpoint recipes
  are identical. G3/G4's 410M clean adaptation used 10M tokens and 2,441 steps
  versus G6's 5M and 1,220; G6-only results are primary and those pools are
  secondary consistency checks.
- Primary G6 whole-table dependence is 16/16 at 410M and 15/16 at 1.4B.
  Primary 410M history specific necessity is 15/16; history sufficiency is
  11/16 and does not pass the registered typical-route rule.
- Pooled 410M whole-table sufficiency and history sufficiency are both 15/21
  with Wilson lower 0.500436. Both are explicitly labeled one-seed knife-edges.
- G6 source plus replay used 12.986 GPU-hours; conservative cumulative allocation is 22.65 GPU-hours.

## Reviewer-requested S2e quality audit

- Exact pre-write clean checkpoints were evaluated on S2e's same clean-token
  slice and paired with verified post-write NLL.
- Mean changes are -1.00e-5 nats at 410M and +1.21e-5 at 1.4B; perplexity
  ratios are 0.999990 and 1.000012.
- The result is labeled post hoc and descriptive. It supports a constructive
  statement that surgical row writes are deletable with no material clean-
  quality cost at this assay's resolution.
- Artifact SHA-256:
  `2c9708c98d39f5780f252814ea18f5c9acc730e3b6149715cfd00c204917da95`.

## Manuscript and artifact audit

- final title: *Addressing Is Not a Security Boundary: Optimizer Policy Moves the Storage Component, Not the Deletion Boundary*
- final manuscript: 13 pages total; references start on page 10 and the appendix starts on page 11, leaving 9 main-text pages
- all 13 pages rendered and visually inspected
- no clipping, overlap, broken table, or unreadable figure found
- abstract: approximately 191 words
- citation audit: 15/15 bibliography entries used; no missing keys; Hase et al. (NeurIPS 2023) is included and explicitly delimits novelty
- relevant CPU tests: 16 passed
- final PDF SHA-256: `897dfbd88805ea97c63c3b13c392b250a9c49e0c11fd9223a42ac7912b8c2ba6`
- sanitized release is restricted to Memory Graft files, contains all generated
  TeX inputs, validates every manifest digest, and has no matched local user
  path, remote root, GPU IP, Hugging Face token, or private-key marker

The release is rebuilt after this audit is added. Its final digest is written to
the adjacent `output/release/SHA256SUMS.txt` rather than embedded here, which
avoids a self-referential archive hash.
