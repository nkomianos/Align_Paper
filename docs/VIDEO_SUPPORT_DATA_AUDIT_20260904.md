# Video removal: support versus synthesis — metadata audit

Status: CPU-only feasibility investigation, not a paper greenlight or model result.

Question: with the same frozen video generator, how much removal error is due to
where edits are permitted, versus reconstructing missing content? Compare object
support, effect-inclusive support, and equal-area dilation, while measuring
unrelated-region preservation. This is not a claim that existing benchmarks are
unfair; inferring physical aftereffects is deliberately part of their task.

The pinned public BeyondMasks dataset revision
`0774e43de9abc576e389c78a319b4a5ff3472fdf` lists 180 complete input/reference/object-mask
triples, 544 files, 2,807,857,340 bytes, and CC-BY-4.0 licensing. Only metadata and
source were fetched. Fourteen receipt hashes were rechecked locally. No videos or
weights have been fetched by this audit. Its dataset card calls the masks object
masks; this does not establish effect-region annotations or pixel-exact paired
counterfactuals. Those remain prerequisites for the proposed comparison.

Source inspection pins DiffuEraser at
`8e6f279ac7531e27ad1849c6f8dab5372a8597e7`. A synthetic compositor check confirms
that the inspected mask transformation leaves pixels outside its editable support
unchanged. This is a code contract, not evidence of actual benchmark invocation,
model weakness, effect localization, or novelty. The microtest passed locally.

Next: inspect a small predetermined sample of the released videos, establish
alignment and independently label eligible effect regions before model outputs.
Do not use raw reference-minus-input differences as effect ground truth without
validating scene alignment. Only then consider a frozen small inference comparison.

Local receipts: `artifacts/beyondmasks_contract_audit_20260904_v1`.
Reproducible metadata collector: `scripts/audit_beyondmasks_contract.py`.
Public sources: https://huggingface.co/datasets/yigitekin/BeyondMasks and
https://github.com/lixiaowen-xw/DiffuEraser .

## Executed sample inspection

Fetched the three smallest triples by total released byte size (175, 168, 171),
not by model performance. All nine files match their pinned LFS SHA-256 values.
Decoded frame counts and dimensions agree within each triple: 20, 38, and 68
frames respectively. This is a size-biased feasibility sample, not representative
evaluation. Script: `scripts/inspect_beyondmasks_samples.py`; raw videos, receipts,
and first/middle/last contact sheets: `artifacts/beyondmasks_samples_20260904_v1`.
Contact sheets resize for inspection and are not measurement inputs.

Visual inspection finds a red moving object and mirror reflection (175), a glass
and tabletop reflection (168), and a lamp with broad illumination changes (171).
The lamp pair differs by about 50/255 mean absolute channel intensity outside
the object mask; about 80% of those pixels exceed a mean difference of 10.
This can be the intended physical illumination effect, NOT evidence that the
pair is invalid. In the other two examples the corresponding fraction is about
1–3% at the inspected frames. Neither statistic identifies true effect support.

This changes the prospective design: illumination and localized reflections must
be separate strata. An equal-area expansion control can be useful for localized
effects, but global illumination may require nearly full-frame editable support.
Do not call unchanged pixels outside an object mask a generic preservation win.
Frame-count agreement also does not establish pixel-exact temporal alignment.

Next qualification: independent effect-region annotation on a prospectively
selected reflection subset, with background alignment checks and uncertainty
regions excluded from pixel-based scoring. A lamp/global-lighting arm needs a
separate assessment protocol. No model has been run and no paper claim is yet
supported by these samples.
