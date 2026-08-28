# Interpretation guard for the recency-gated G0

## Status

This note does **not** modify the frozen G0 configuration, its running code, or
its pass/fail decision.  It records a post-G0 validity requirement for any
subsequent paper claim.

## Why a successful single-recipe gate is insufficient

[The Model Organism Lottery](https://arxiv.org/abs/2607.01033) reports that
white-box interpretability outcomes in language-model model organisms vary
substantially with the model-organism training methodology, even after behavior
expression is controlled.  In particular, integrated training can be less
interpretable than familiar post-hoc fine-tuning.  This is directly relevant to
G0: its sequential LoRA model organism is a useful causal feasibility test, but
cannot on its own establish a general mechanism of alignment faking or a robust
mitigation.

## Post-run necessity audit required

The frozen G0 runner records both its signed steering and erasure quantities
without applying the matching intervention to the cue-only control.  That is
not a complete difference-in-differences control: either intervention could
change generic sensitivity to the monitored/unmonitored headers in both models,
rather than only the learning-induced policy effect.  The frozen run must finish
unchanged, because its artifacts remain valuable diagnostic evidence, but its
reported mediation and necessity rows cannot green-light the candidate.

After retrieval verification, a separate immutable analysis must apply the
same signed steering or erasure direction (and every matched control) to both
the baseline and cue-only adapters, and calculate the corresponding
difference-in-differences quantity, for example:

```
(baseline - cue-only) - (baseline_erased - cue-only_erased)
```

The audit writes to a fresh output root and does not change G0 metrics,
evidence, or decision.  Its result is descriptive only: even a favorable
corrected estimate can authorize at most a freshly preregistered G1, never a
retroactive G0 pass.

## Non-negotiable expansion condition if G0 passes

Before calling the recency mechanism a viable ICLR paper, the next protocol
must pre-register and test all of the following:

1. **Training-recipe replication.**  Replicate the directional readout,
   signed steering/erasure specificity, and temporal-homogenization result
   under at least one materially different post-training recipe.  It must not
   merely change a learning rate or seed.
2. **Model-family replication.**  Run the same preregistered causal analyses
   on an independent model family.  A successful Qwen-only result is a finding,
   not a general paper claim.
3. **Integrated, not only post-hoc, setting.**  Include a protocol where the
   contextual policy is integrated with unrelated post-training data rather
   than appended as a narrow final behavioral fine-tune.
4. **Held-out causal testing.**  Preserve the existing split discipline: select
   the timestamp direction and layer on one partition, and assess causal and
   behavioral quantities only on disjoint aliases/templates.
5. **Fail-closed result.**  If the direction or mitigation is recipe-specific,
   describe G0 as a negative external-validity result and do not tune a new
   recipe until it passes by a separately declared protocol.

## Scientific consequence

G0 remains worth running because it can quickly falsify the strongest proposed
mechanism.  But a pass authorizes **robustness testing**, not a paper claim,
and this guard prevents a misleading conclusion from an easy-to-interpret
single training construction.
