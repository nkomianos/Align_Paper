# EndoPAHF Neural Transfer G2 V2 Protocol

Status: prospective full-learning repair, frozen before any capable EndoPAHF
training output. Synthetic Qwen3.5-9B gradient G0 and policy G1 remain mandatory
prerequisites. Exact-interface preflight must also qualify.

## Question

On public PAHF shopping tasks for previously observed users, can sparse delayed
probes distinguish transient expression from persistent preference transition,
and can paired delayed-minus-immediate SDPO improve the old persistent target
more than spending the same delayed labels alone?

## Repair and fixed training design

The superseded G2 v1 used 128 learning bases and four redundant rotations of
each in its global updates. A development-only, position-invariant diagnostic
found the old target at 27.08% accuracy and NLL 1.8526. With all 630 public
learning bases it reaches 47.92% and 1.0744. No capable endpoint existed, so G2
v2 prospectively replaces the design rather than interpreting a failed run.

V3 supplies 630 learning bases under four rotations. For each arm, a fixed
outcome-blind schedule chooses exactly one balanced rotation from every base:
630 unique bases, 18 examples per batch, 35 updates. Evaluation still averages
all four rotations within each base. This increases semantic/user-feature
coverage without paying for four nearly duplicate population updates.

Four disjoint outcome-blind panels contain 16 base tasks each, retaining 64
distinct anchor bases in total. Each anchor step uses both selected bases under
all four rotations (eight rows). The required arms are baseline, raw immediate
SDPO, full delayed-expression oracle, transition sanity, and for every panel:
delayed-anchor-only SDPO, delayed-anchor-only SFT, and population immediate plus
the paired delayed-minus-immediate residual. The four panel models are averaged
in probability space; no panel is selected from outcomes.

There are 15 independently reset trained arms and 35 steps per arm (525
optimizer updates), versus 864 in the retired design. Every checkpoint,
optimizer, step ledger, schedule and prediction grid is preserved.

## Gates

The estimand and confirmation inference remain unchanged. DEV has 96 bases and
routes onward at mean paired old-target NLL gain at least .03 with accuracy
noninferiority. It cannot establish a result. Confirmation has 256 bases and
requires gain at least .05, a positive lower endpoint from the 10,000-resample
base-cluster bootstrap, and accuracy noninferiority within .02.

Augmented must beat equal-label anchor SDPO, anchor SFT, and raw immediate. Raw
must acquire the new target; the oracle must acquire the old target; opposed
directions must separate. The transition sanity adapter and predictions must
match raw within `1e-6`. Choice mass and per-label position controls remain.

The revised DEV routing audit is checksum-verified: null routing `.2063`,
planned-signal power `.9159`, and noisy-signal power `.8357`. Evidence root is
`artifacts/hindsight_endo_pahf_g2_v2_power_20260904_v2`; manifest SHA-256 is
`8e050034619ff21fddfbc73bb337cdee0f41c4a0117cb816b8c8100ac935e4ca`.
This validates only the routing rule, not the hypothesis.

The v1 audit has identical rates but is preserved as superseded because its
receipt omitted the transitive scoring-module hash.

The capable preflight and G2 runners now bind V3 manifest
`2ae32c119087d97de6f2e5a65959b4c94a7d81362732871ff806b157e000fab2`.
Confirmation remains unreadable to its runner until DEV independently verifies
and qualifies. A confirmation pass would still require another model family and
real longitudinal or human validation before a paper green light.
