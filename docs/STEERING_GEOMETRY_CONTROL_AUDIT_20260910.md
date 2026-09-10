# Steering candidate: source-control audit

**Decision:** no replication launch from this audit. No new paper result.

[Paper v1](https://arxiv.org/html/2609.06289v1), inspected methods and appendices
F/J/K: the authors already test paraphrasing, disjoint samples and single-label
subsets. Transfer uses row/column residualization, and geometry is compared with
target accuracy as a predictor. These overlap the proposed generic robustness
question. Correlation comparison is not identical to prospectively matching
realized intervention strength, but that distinction alone is insufficient novelty.

Pinned [author release](https://github.com/DeepRCL/Steering_Geometry/tree/6ebdd0b9887fbd2d7ee1692de9aabe5e7c1c1666)
and saved three source files with SHA256 receipts in
`artifacts/steering_geometry_source_20260910`. Reproduction README requires local
datasets and says a Hugging Face release link is pending. The transfer metric
code calls ordinary Spearman correlation on flattened pair entries.

Independent assessment: pairs sharing source/target values need dependence-aware
inference; a value-label permutation analysis is a possible check. This does not
establish a false correlation, invalidate the paper, or supply our contribution.
Raw transfer matrices were not obtained here. No substitute synthetic dataset,
unreviewed upstream execution, or model download was used to claim replication.
