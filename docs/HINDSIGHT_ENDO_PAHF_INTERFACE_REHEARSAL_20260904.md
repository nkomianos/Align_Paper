# EndoPAHF CPU Interface Rehearsal

Status: prospectively frozen apparatus check. This is not a scientific gate and
cannot green-light the paper.

## Question

Can the cached Qwen3-0.6B model reliably map explicit PAHF preference statements
to the intended A/B/C/D action, while assigning enough first-token probability
mass to those four labels for later full-vocabulary objectives to be meaningful?

This check deliberately does **not** train a policy, test endogenous feedback,
or establish that real users exhibit preference transitions. It only catches a
broken task serialization, label-token problem, or incapable small-model
interface before renting another accelerator.

## Frozen inputs and selection

- Input: the sealed development split in
  `artifacts/hindsight_endo_pahf_external_20260904_v1`.
- Model: `Qwen/Qwen3-0.6B` at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, loaded locally and offline.
- Selection: the first 32 development records after sorting by SHA-256 of
  `record_id | endo-pahf-interface-rehearsal-v1`.
- No confirmation examples are opened.
- The four choice labels must each be one native tokenizer token.

For every selected record, the model receives an explicit extraction query in
three conditions: immediate feedback (new target), delayed expression feedback
(old persistent target), and delayed transition feedback (new persistent
target). Scores use full-vocabulary next-token log probabilities; A/B/C/D are
normalized only for choice accuracy and target probability, while their total
mass under the full vocabulary is separately retained.

## Prospective apparatus qualification

The interface qualifies only if all three conditions satisfy:

- at least 28/32 normalized-choice predictions are correct;
- mean normalized target probability is at least 0.70; and
- mean full-vocabulary A/B/C/D mass is at least 0.10.

Failure means only that this small-model prompt/interface is unsuitable. Pass
means only that the external neural assay can use this serialization. Neither
outcome is evidence for or against the Hindsight paper thesis.
