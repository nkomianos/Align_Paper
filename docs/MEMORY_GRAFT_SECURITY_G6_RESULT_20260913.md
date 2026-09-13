# G6 result: optimizer dose, quality cost, and route distribution

**Status: valid confirmatory result. The 644-file source manifest validates,
and full deterministic replay reproduces the decision, all scientific metrics,
and every row in 320 prediction files exactly.**

## Dose response

G6 ran 16 prospectively fixed new seeds per Pythia size at four paired table
AdamW rates. The backbone rate remained `5e-5`; poison examples and dropout RNG
were paired within seed. All 128 dose cells passed installation eligibility and
all cells remain in every prevalence denominator.

Whole-table necessity above the registered 0.15 meaningful-effect scale occurs
in `0/16, 2/16, 15/16, 16/16` new Pythia-410M seeds at table rates
`5e-5, 1e-3, 1e-2, 1e-1`. The corresponding Pythia-1.4B counts are
`0/16, 0/16, 2/16, 15/16`. Outside-table sufficiency moves oppositely:
`16/16, 16/16, 2/16, 0/16` at 410M and `16/16, 16/16, 15/16, 1/16` at 1.4B.
The optimizer-route transition is therefore sharp within each size and occurs
at a higher table rate in the larger model.

The first table-dependent rate among new 410M seeds is `1e-3` for 2/16,
`1e-2` for 13/16, and `1e-1` for 1/16. At 1.4B it is `1e-2` for 2/16,
`1e-1` for 13/16, and absent for 1/16. This is a seed distribution, not evidence
for an unobserved discrete latent mechanism.

At the fixed `1e-1` endpoint, the primary new-seed counts are:

| Route indicator (>0.15) | 410M prevalence (Wilson 95%) | 1.4B prevalence (Wilson 95%) |
|---|---:|---:|
| whole-table necessity | 16/16 [0.806, 1.000] | 15/16 [0.717, 0.989] |
| whole-table sufficiency | 11/16 [0.444, 0.858] | 13/16 [0.570, 0.934] |
| outside-table sufficiency | 0/16 [0.000, 0.194] | 1/16 [0.011, 0.283] |
| final-row necessity | 9/16 | 13/16 |
| final-row sufficiency | 7/16 | 7/16 |

Thus whole-table dependence is robust in the prospective cohort. Whole-table
sufficiency passes the registered `typical` rule at 1.4B, but not at 410M.
Pythia-1.4B seed 03 has whole-table necessity 0.149414, only 0.000586 below the
frozen 0.15 indicator threshold, and remains classified as absent.

## Pooled fixed endpoint

Pooling the 16 new seeds with the five prospectively declared verified parent
seeds gives 21 runs per size at table rate `1e-1`. The 410M parent seeds are the
valid G3 `AdamW, lr=1e-1, weight_decay=0.01` arm; the 1.4B parents are the
separately fixed G3.1 arm with that same endpoint. Model revisions, clean-token
source offsets, graft, markers, poison count, poison schedule, batch size,
optimizer rates and decay, and evaluation match G6. G3/G4's 410M clean
checkpoints used 10M adaptation tokens and 2,441 steps, whereas G6 used 5M and
1,220 steps. The 1.4B G3.1/G6 clean recipe is identical. The invalid original
G3 1.4B selection is not pooled. The new G6 cohort is primary; the declared
pool is a secondary consistency estimate, and the 410M portion is heterogeneous
in clean-adaptation length.

| Route indicator (>0.15) | 410M prevalence (Wilson 95%) | 1.4B prevalence (Wilson 95%) |
|---|---:|---:|
| whole-table necessity | 21/21 [0.845, 1.000] | 20/21 [0.773, 0.992] |
| whole-table sufficiency | 15/21 [0.500, 0.862] | 17/21 [0.600, 0.923] |
| outside-table sufficiency | 0/21 [0.000, 0.155] | 1/21 [0.008, 0.227] |
| final-row necessity | 12/21 [0.365, 0.755] | 17/21 [0.600, 0.923] |
| final-row sufficiency | 10/21 [0.283, 0.676] | 9/21 [0.245, 0.635] |

The whole table is typically necessary and sufficient at both sizes under the
registered definition (Wilson lower bound greater than 0.5), but 410M
whole-table sufficiency is a knife-edge call: its lower bound is 0.500436, only
0.000436 above the bar, and one seed flip would reverse it. The 21/21 necessity
result is robust to one flip. `Typical` means a majority of training runs and is
far weaker than a security guarantee; 15/21 observed successes cannot license
per-item deletion. The 410M pool is additionally interpreted only as a
cross-clean-adaptation consistency check. The nominal final
rows are not a reliable self-contained item boundary: their sufficiency is not
typical at either size, and their necessity is not typical at 410M.

## Paired quality cost

Post-minus-pre clean NLL rises with the strongest table rate.

| Rate | 410M mean [bootstrap 95%] | 1.4B mean [bootstrap 95%] |
|---|---:|---:|
| 5e-5 | 0.00016 [-0.04017, 0.02671] | 0.00748 [0.00588, 0.00906] |
| 1e-3 | 0.00297 [-0.03397, 0.02587] | 0.00821 [0.00583, 0.01072] |
| 1e-2 | 0.01456 [-0.03370, 0.04710] | 0.01227 [0.01028, 0.01416] |
| 1e-1 | 0.07732 [0.05139, 0.11505] | 0.06894 [0.06691, 0.07096] |

At `1e-1`, these changes are 2.66% and 2.64% of the respective mean pre-poison
NLL and imply geometric-mean perplexity ratios of 1.0804 and 1.0714 (8.04% and
7.14% increases).

Matched-benign post-minus-pre accuracy at `1e-1` is 0.97687 at 410M and
0.99994 at 1.4B. These outcomes are reported descriptively; G6 registered no
quality-preservation gate.

## Temporal route distribution

At 410M and `1e-1`, the primary 16 new G6 seeds give:

| Route indicator (>0.15) | Prevalence (Wilson 95%) |
|---|---:|
| 72-row history specific necessity | 15/16 [0.717, 0.989] |
| 72-row history sufficiency | 11/16 [0.444, 0.858] |
| earlier-row necessity | 10/16 |
| final-row necessity | 9/16 |
| final-row sufficiency | 7/16 |
| shared-prefix control drop | 4/16 [0.102, 0.495] |

The declared secondary pool with five G4 seeds gives:

| Route indicator (>0.15) | Prevalence (Wilson 95%) |
|---|---:|
| 72-row history specific necessity | 19/21 [0.711, 0.973] |
| 72-row history sufficiency | 15/21 [0.500, 0.862] |
| earlier-row necessity | 13/21 [0.409, 0.792] |
| final-row necessity | 12/21 [0.365, 0.755] |
| final-row sufficiency | 10/21 [0.283, 0.676] |
| shared-prefix control drop | 5/21 [0.106, 0.451] |

The broad, input-computable history footprint is typically specifically
necessary in the primary cohort but is not typically sufficient. The pooled
15/21 sufficiency call has the same 0.500436 lower bound and one-flip fragility,
and the G4/G6 pool mixes 10M- and 5M-token clean adaptations. Its allocation
between earlier and final rows varies across seeds. The
shared-prefix control produces large drops in some seeds, so this result does
not establish a universally isolated per-item boundary.

## Licensed interpretation

Optimizer policy causally changes whether the behavior is stored in the
addressable component, with a scale-dependent transition and a measurable clean
NLL cost at the strongest rate. Component routing does not by itself determine
a reliable nominal item boundary. The result does not establish generality to
jointly pretrained conditional-memory models, semantic payloads, or arbitrary
optimizers.

## Provenance

- source artifact: `artifacts/memory_graft_security_g6_run1`
- source manifest entries: 644
- source manifest SHA-256: `ae40a13c11dc1932d3b0e4965c468c4868aff6a73b150d7efc626f8f10baa588`
- source decision SHA-256: `7091c21ff39317fb7c544079fac099f6ee2d8b5f0c6e09f7cf80003afc5e1c08`
- source runner wall time: 23,374.50 seconds
- full-replay report: `artifacts/memory_graft_security_g6_verification.json`
- full-replay SHA-256: `55fc4c77a07571e4e16e02c9e1fdadde714d16224d6f7457ec4da4d55af98fd0`
- replay decision exact: yes
- maximum scientific metric absolute difference: 0
- prediction-row disagreements: 0 across 320 files
- source plus replay runner time: 12.986 hours
