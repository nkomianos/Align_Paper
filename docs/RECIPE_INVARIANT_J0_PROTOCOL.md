# J0 protocol: recipe-invariant causal mechanism selection

## Status

**Prepared and tested; not launched.**  This protocol is a successor candidate,
not an interpretation of the active recency G0.  It must not run until the
active gate has been retrieved and integrity-verified and the literature audit
has confirmed that the precise held-out selection claim is still distinct.

## Claim under test

Train the same harmless two-action contextual policy through three materially
different post-training recipes:

1. post-hoc SFT (recipe A);
2. a contrastive preference objective (recipe B); and
3. SFT integrated with an equal mass of unrelated routing examples (recipe C).

Only A and B may be inspected to select one residual direction and layer.  The
predeclared selection score is cross-recipe direction agreement on the train
aliases.  Recipe C, and all held-out aliases, remain unavailable to selection.
On recipe C held-out aliases, compare the selected direction's signed steering
and projection-erasure effect with equal-norm random, principal-component, and
each single-recipe direction.  This is a selection-generalization experiment,
not a claim that a high activation correlation is causal.

## Gate

Each of two seeds must simultaneously show:

- recipe-C signed steering at least 8 percentage points, with a 95% bootstrap
  lower bound of 4 points;
- at least 30% recipe-C effect reduction under erasure, lower bound 15%;
- every equal-budget control no larger than 40% of the selected direction's
  corresponding effect;
- at most 5 percentage points ordinary-behavior loss; and
- a steering margin of at least 2 points over every equal-budget baseline.

Any failed row kills the candidate.  Passing only licenses an offline,
revision-pinned reproduction, then replication on a second backbone and a
separately generated behavior family.  It is not grounds for a paper claim.

## Implementation and checks

- Frozen contract: `configs/recipe_invariant_mechanisms_j0.yaml`
- Fail-closed corpus builder and gate: `src/recipe_invariant_mechanisms/gate.py`
- CPU protocol tests: `tests/test_recipe_invariant_mechanisms.py`

The corpus contains only deterministic nonce aliases and `ALPHA`/`BETA`
routing labels.  Its train/held-out split and recipe-C exclusion are structural
invariants checked before accelerator work is authorized.
