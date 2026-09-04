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
