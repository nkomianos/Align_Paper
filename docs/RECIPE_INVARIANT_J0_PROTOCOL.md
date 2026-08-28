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
aliases, computed from each adapter's hidden-state update relative to that
same adapter-disabled base model.  This subtraction is required so that a
shared lexical difference between `TARGET_MODE_A` and `TARGET_MODE_B` cannot
be selected as though it were a learned mechanism.  Recipe C, and all
held-out aliases, remain unavailable to selection.
On recipe C held-out aliases, compare the selected direction's signed steering
and projection-erasure effect with equal-norm random, principal-component, and
each single-recipe direction.  This is a selection-generalization experiment,
not a claim that a high activation correlation is causal.

The signed steering outcome is the paired change in the **B-versus-A response
gap** under positive versus negative steering.  A direction that merely raises
the probability of one routing token in both contexts cannot pass.  Point
estimates and paired bootstrap lower confidence bounds are recorded in distinct
fields; this pre-run repair corrects an unlaunched draft that had confused
bootstrap endpoints with the point estimate.  The corpus, recipe split,
thresholds, controls, and held-out C protocol remain fixed.

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
- Runnable training, selection, and recipe-C evaluation: `src/recipe_invariant_mechanisms/runner.py`
- Offline artifact verifier: `src/recipe_invariant_mechanisms/verify.py`
- CPU protocol tests: `tests/test_recipe_invariant_mechanisms.py`

The corpus contains only deterministic nonce aliases and `ALPHA`/`BETA`
routing labels.  Its train/held-out split and recipe-C exclusion are structural
invariants checked before accelerator work is authorized.

The J0 configuration pins the exact public model snapshot already cached on
the GH200.  The launcher forces Hugging Face offline mode, so a future run
cannot silently refresh model metadata or weights.

After completion, retrieval must use `scripts/retrieve_recipe_invariant_j0.sh`.
It verifies the run/config/corpus/protocol digests, every immutable adapter
file, and that the recorded selection was made from A/B before C was opened.
