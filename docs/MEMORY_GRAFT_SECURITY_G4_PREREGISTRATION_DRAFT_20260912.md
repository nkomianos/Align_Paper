# G4 draft: temporal row footprint of a table-dependent write

**Status: draft, not frozen, and not evidence.** G4 is motivated by the verified
G3 410M result. No G4 model has been loaded or trained.

## Question

G3 shows that aggressive table optimization can make the complete table
necessary and sufficient while the 16 rows addressed at the trigger's final
position remain unreliable. G4 asks whether the missing behavior occupies the
other deterministic hash rows read at earlier positions inside the trigger.
This distinguishes a wider but still item-computable temporal footprint from
diffuse context-dependent or co-adapted table storage.

## Fixed source and training

Use only Pythia-410M and the five sealed S1 clean checkpoints at seeds
26091301--26091305. Repeat G3's fixed N=64 joint training with backbone AdamW
learning rate 5e-5, table AdamW learning rate 0.1, weight decay 0.01, 512 steps,
length 256, and effective batch 16. There is no development or profile
selection. The 1.4B G3 result is excluded because its selection did not
reproduce.

## Prospectively defined row sets

Before loading weights, compute addresses from token IDs, frozen compression,
hash seed, and table layout:

- `final`: valid hash rows read at the final trigger token;
- `earlier_internal`: valid hash rows at earlier trigger positions whose entire
  n-gram window lies within the trigger string;
- `all_internal`: the union of `final` and `earlier_internal`.

Any exact-bank hit at a registered internal position invalidates the apparatus,
because that position would bypass the trainable hash table. Construct the same
three sets for the benign marker. Construct 16 random controls matched exactly
to `all_internal` in row count per hash table, excluding trigger and benign
rows. Record all row IDs and overlaps before training.

## Interventions and estimands

For every trained checkpoint, measure intact and clean ASR, then independently
restore from the clean checkpoint: final rows, earlier-internal rows,
all-internal rows, benign all-internal rows, each random matched set, and the
whole table. Also transplant final, all-internal, and whole-table learned rows
into the clean checkpoint. Every intervention starts from an endpoint checkpoint.

The primary estimand is specific all-internal necessity:

`(intact - all_internal_restored) - max(benign_drop, random_drops)`.

Secondary estimands are all-internal sufficiency, whole-table necessity and
sufficiency, outside-table sufficiency, final-row necessity and sufficiency,
and the incremental earlier-row effect
`final_restored_ASR - all_internal_restored_ASR`. Each is reported separately;
there is no compound gate.

## Derived thresholds

The smallest scientifically meaningful ASR effect remains 0.15. With 1,024
paired predictions, the registered distribution-free 95% half-width is
`sqrt(2 ln(2/0.05)/1024) = 0.0848813`. The installation apparatus check therefore
requires intact-minus-clean ASR at least 0.2348813 in each seed; failure stops
interpretation for that seed but is reported. A scientific estimand passes only
when its two-sided five-seed Student-t lower endpoint exceeds 0.15. The
Student-t interval estimates a mean seed-level contrast and remains unclipped.

## Interpretation

- Positive specific all-internal necessity with positive incremental earlier
  effect locates the G3 nominal-row gap in the trigger's temporal address
  footprint.
- Positive all-internal necessity without incremental earlier effect means the
  larger set helps but does not specifically implicate earlier rows.
- Whole-table necessity with failed all-internal necessity means deterministic
  item-internal addresses are still insufficient for deletion; context rows or
  table/backbone co-adaptation remain candidates.
- Failed whole-table necessity fails to reproduce G3's component route and
  blocks row-footprint interpretation for that seed set.

## Scale and verification

Five source runs and five complete replays are projected below 0.5 GPU-hours,
under 1% of the approximately 42.5 hours remaining. This scale is justified by
using the same five independent training units as G3, 1,024 paired prompts, all
prospectively defined row controls, and exhaustive replay of the only verified
scale. It is not chosen merely because it is cheap. Freeze code, config, this
document, source manifests, and row inventories before training. Any registered
decision mismatch under full replay invalidates that claim.
