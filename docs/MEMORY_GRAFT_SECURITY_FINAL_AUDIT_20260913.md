# Final memory-boundary paper audit — 2026-09-13

## Outcome

The manuscript is scientifically defensible and ICLR-caliber. The evidence supports submission; it does not support a guarantee of acceptance. The remaining scientific limitations are explicit: synthetic one-token probes, retrospectively grafted rather than jointly pretrained memory, and two model families. The second review round also distinguishes primary G6 estimates from heterogeneous 410M/G4 parent pools.

A subsequent external review raised the possibility that the graft was
functionally vestigial. A frozen checkpoint-only audit rejects that explanation:
bypassing the trained graft worsens held-out clean NLL in 6/6 410M and 5/6
1.4B clean checkpoints. This supports functional use, not a separately trained
ungrafted-model comparison. The paper now states explicitly that retrofitting a
graft after dense pretraining can still bias later optimization toward mature
backbone circuits.

G7 subsequently attempted the highest-value structural follow-up at
approximately 167M parameters and 1.0066B tokens. It produced a load-bearing
jointly trained memory path, but neither matched architecture installed the
registered storage probe. The routing estimand is invalid. The complete replay
agrees on that decision but fails the frozen tensor/prediction exactness rule,
so even the load-bearing and NLL observations remain developmental. G7 is
disclosed in the limitations and evidence table and does not alter the final
claim or acceptance calibration.

The separately preregistered G8 successor reused the six G7 source checkpoints
and changed to a frequent-token, zero-occurrence trigger. It failed the derived
unconditional-learnability calibration at every registered exposure and step
cell, with a maximum six-checkpoint gain of 0.0009766 against a 0.15 gate. All
48 completed short runs replay exactly. G8 therefore ran no conditional routing
cell and supplies no routing evidence.

G9 made one final, separately frozen repair using a common payload, matched
training/evaluation prediction position, a three-rate sweep, and continuous
payload instrumentation. All six checkpoints still missed both calibration
gates. The strongest rate improved payload log-probability by 4.13--4.52 nats
and median rank to 117--145, but exact gain remained zero and MRR gain stayed
below 0.011. All 18 source/replay pairs were bitwise exact. This is a valid
capability-calibration negative; routing was never run and the joint-pretraining
repair line is now permanently closed.

G10 then attempted the single preregistered semantic-payload successor on the
mature retrofitted 410M apparatus. Its unconditional three-token calibration
missed the 0.15 sequence-exact gate with zero gain, so no semantic routing or
deletion cell ran. Aggregate teacher-forced MRR improved by 0.613, but the
first token remained top-1 in 0/1,024 contexts and later-token gains depended
on supplying the correct prior token. Source/replay are bitwise exact. G10 is a
valid negative capability calibration and leaves the semantic limitation open.

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
- final manuscript: 14 pages total; references start on page 10, leaving 9 main-text pages
- all 14 pages rendered and visually inspected
- no clipping, overlap, broken table, or unreadable figure found
- abstract foregrounds mapping capacity at both sizes and the 8.04%/7.14% aggressive-optimizer quality cost; the asymmetric 5.57%/0.14% clean contribution remains in Section 4.1
- citation audit: 15/15 bibliography entries used; no missing keys; Hase et al. (NeurIPS 2023) is included and explicitly delimits novelty
- relevant CPU tests: 22 passed
- final PDF SHA-256: `f01457a25452dca2a5e534ac83658e582a23d9f215fd788f988ac665e15ba6c0`
- sanitized release is restricted to Memory Graft files, contains all generated
  TeX inputs, validates every manifest digest, and has no matched local user
  path, remote root, GPU IP, Hugging Face token, or private-key marker

## G7 closure audit

- all six source and six replay pretrains completed at the registered token count
- all twelve posttraining invocations completed after a narrow pre-outcome
  marker-parser amendment
- all 24 run manifests validate internally
- installation failed in every ordinary and frozen-component seed in both arms
- source and replay categorical decisions are identical, but all checkpoint
  pairs fail tensor exactness and 135,974/221,184 raw prediction rows differ
- G7 source plus replay used 18.098 measured GPU-hours; cumulative use including
  the excluded benchmark is approximately 42.283/50 hours
- no G7 routing, row-locality, or general jointly pretrained-memory claim enters
  the abstract, results, or conclusion
- result memo: `docs/MEMORY_GRAFT_SECURITY_G7_RESULT_20260914.md`

## G8 closure audit

- the frequent-token trigger's constituents occur 11,379--67,393 times in the
  frozen pretraining stream; the full phrase occurs zero times
- all six inherited checkpoints fail all four unconditional calibration cells
- maximum exact-match gain is 1/1,024 in one dense seed; all other gains are zero
- all 48 completed short runs are state-hash and prediction exact across replay
- no conditional adaptation, component transplant, or row intervention ran
- G8 used approximately 0.382 GPU-hours including two stopped pre-outcome
  launches; cumulative use is approximately 42.665/50 hours
- result memo: `docs/MEMORY_GRAFT_SECURITY_G8_RESULT_20260915.md`

## G9 closure audit

- common payload frequency: 67,393 occurrences in the frozen pretraining stream
- causal prediction position matched exactly between training and evaluation
- 36/36 registered calibration invocations completed; every post-calibration
  checkpoint and every-step continuous trace retained remotely
- no exact-match gate passed; strongest-rate exact gain was zero in all six
- strongest-rate log-probability gain was 4.130--4.515 nats, median rank
  117--145, and MRR gain 0.006642--0.010443 versus the 0.15 criterion
- frozen classification: capability failure; routing, transplant, and deletion
  cells never run
- 18/18 source/replay pairs reproduce exactly
- G9 used 0.2791 GPU-hours; cumulative measured use is about 42.944/50 hours
- result memo: `docs/MEMORY_GRAFT_SECURITY_G9_RESULT_20260915.md`

## G10 closure audit

- mature retrofitted Pythia-410M clean checkpoint and graft reused from S1
- unconditional three-token semantic mapping trained for 512 exposures/steps
  with matched training and evaluation prediction position
- sequence exact-match gain: 0 against the registered 0.15 gate
- aggregate teacher-forced MRR gain: 0.6130; first-token MRR gain: 0.00548
- first-token top-1: 0/1,024 before and after; later-token top-1 after training:
  86.1% and 99.5% when correct predecessors are supplied
- no conditional training, component transplant, or row-deletion cell ran
- source/replay reports, logs, state hashes, predictions, and manifests match
  exactly
- G10 used 0.04730 GPU-hours; cumulative measured use is about 42.991/50 hours
- result memo: `docs/MEMORY_GRAFT_SECURITY_G10_RESULT_20260915.md`

The release is rebuilt after this audit is added. Its final digest is written to
the adjacent `output/release/SHA256SUMS.txt` rather than embedded here, which
avoids a self-referential archive hash.
