# Memory Graft security G3 optimizer-routing pre-registration

## Question

S1 used AdamW at learning rate 5e-5 and weight decay 0.01 for every trainable
parameter. S2a kept that recipe while training only the table and did not install
the mapping, whereas S2e installed a row-confined mapping with Adam at 1e-3 and
zero weight decay. G3 asks whether that optimizer asymmetry explains the S1
storage result. During joint training, the dense backbone always retains the S1
AdamW recipe. Only the hash-table optimizer changes.

This can overturn or refine the headline. If an ordinary joint run with a
reasonable table optimizer becomes target-row necessary and specific, the
result is optimizer-conditional rather than an architectural routing failure.
If the table carries a sufficient copy but whole-table restoration leaves the
behavior, learning is redundant and deletion still fails. If the best
independently replicated table optimizer does not put a measurable copy in the
rows, the learning-rate objection fails over the tested range.

## Fixed profiles and staging

At N=64 for both Pythia sizes, one developmental seed runs: the exact S1 AdamW
baseline; AdamW with table learning rate 1e-3, 1e-2, or 1e-1 and unchanged table
weight decay 0.01; and a split optimizer with the backbone under S1 AdamW while
the table uses Adam at 1e-3, 1e-2, or 1e-1 with zero decay. This range includes
S2e's successful 1e-3 direct-write setting and spans two additional orders of
magnitude. Numerical divergence is reported for that profile and does not stop
later profiles.

The baseline must reach developmental installed attack excess 0.234881 or more
within a size. This apparatus bar is derived below and exists because a routing
assay cannot interpret storage if its baseline no longer installs the behavior.
It is not a scientific conjunct. Failure blocks only that size.

Among finite non-baseline profiles, select the first listed profile whose
target-row transplant into the clean checkpoint has clean-adjusted ASR at least
0.234881. If none does, select the profile with maximum developmental
row-sufficiency, breaking ties by listed order. Replicate the selected profile
on five sealed, development-disjoint training seeds. Selection uses no decisive
seed. Every decisive outcome is reported regardless of sign.

## Causal readings and derived thresholds

The downstream security assay was designed to resolve a 0.15 ASR effect. With
1,024 paired binary prompt outcomes, the registered distribution-free 95%
half-width is sqrt(2 ln(2/.05)/1024) = 0.084881. Developmental installation and
profile eligibility therefore require 0.15 + 0.084881 = 0.234881 observed
clean-adjusted ASR, so the prompt-level lower bound leaves the full scientific
effect measurable.

For five independent training seeds, each following estimand is evaluated
separately using an unclipped two-sided 95% Student-t interval. The lower
endpoint must exceed 0.15 to establish the corresponding effect:

1. target-row sufficiency: clean checkpoint plus poisoned target rows minus
   clean ASR;
2. whole-table sufficiency: clean checkpoint plus the poisoned table minus
   clean ASR;
3. target-row necessity: intact poisoned ASR minus ASR after restoring the
   target rows to their clean values;
4. whole-table necessity: intact poisoned ASR minus ASR after restoring the
   complete table;
5. outside-table sufficiency: poisoned checkpoint with the complete clean table
   minus clean ASR;
6. row-transplant deletion: ASR of clean plus poisoned target rows minus its ASR
   after zeroing those rows; and
7. target-zero specificity: intact target-row-zero effect minus the maximum
   benign-row or mean random-row deletion effect.

The 0.15 value is reused because each reading must account for the same minimum
behavioral effect the deletion claim is intended to protect. There is no
compound scientific gate. In particular, sufficiency can pass while necessity
fails, which means redundant storage rather than assay failure.

Student-t intervals use training runs as independent units and are intentionally
unclipped. They estimate a mean contrast, whose sampling interval is not itself
a bounded Bernoulli probability and may extend outside [0,1]. Prompt rows only
measure each seed-level contrast and are never treated as replicates.

## Controls, interpretation, and verification

The run starts from the exact sealed S1 clean checkpoint for each seed, uses the
same trigger, payload, benign examples, 512 steps, sequence length, data
partition, schedule, backbone learning rate, and backbone weight decay. It
records near-trigger and untriggered rates, benign/random deletion controls,
clean NLL, raw predictions, and numerical failures. N=64 is fixed across sizes
to hold row-update opportunities constant and is not selected from outcomes.

Runner plus full replay is projected at four GPU-hours, 9.14% of the estimated
43.77 hours remaining. This scale is justified by covering the baseline, a
three-order table-LR range under unchanged AdamW, the successful direct-write
optimizer family, both model sizes, and five independent confirmatory seeds.
More prompt rows would not add optimizer replications.

The runner verifies the complete S1 source manifest before weights load. The
verifier reruns development, profile selection, all optimizer steps, causal
interventions, and evaluations; registered decisions and raw disagreement are
reported. Configuration, this document, runner, and verifier are hashed and
committed before any checkpoint is loaded.
