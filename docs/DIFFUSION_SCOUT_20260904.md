# Diffusion inference scout — 4 September 2026

## Decision

Do not open a generic diffusion-uncertainty or order-consistency paper.
The primary-source search below finds substantial prior coverage. A narrower
**cached-verification independence audit** has a checked mathematical witness
and now merits a pretrained-model diagnostic, not a paper or GPU green light.
See [operator audit](CACHE_VERIFICATION_AUDIT_20260904.md).

This is a research triage record, not an exhaustive novelty certification.
Abstract-only checks are sufficient to reject our broad novelty claims, but
not to allege a fault in the cited methods.

## Claims eliminated before model compute

| Proposed broad claim | Primary source checked | Consequence |
| --- | --- | --- |
| Denoising conditionals disagree across orders; local circulation diagnoses it | [Path-Dependent Denoising](https://arxiv.org/abs/2605.09303), abstract | Direct overlap; do not relabel local curl as our contribution. |
| Confidence-driven token selection changes the sampling law | [Demystifying MaskGIT](https://arxiv.org/abs/2510.04525), abstract | Adaptive order and implicit sampling bias already studied. |
| Steered sampling needs correction before estimating semantic uncertainty | [Diversity-steered uncertainty](https://arxiv.org/abs/2510.21310), abstract | Covers autoregressive and masked diffusion generation with reweighting. |
| Confidence stability across denoising steps improves decoding | [DecoCal](https://aclanthology.org/2026.acl-long.545/), abstract | History aggregation and remasking already proposed. |
| Separate parameter uncertainty from diffusion randomness | [FLARE](https://arxiv.org/abs/2602.09170), primary abstract | Broad decomposition is not new; language-specific utility would need evidence. |
| Exact sampling cost depends on unmasking schedule | [Optimal Inference Schedules](https://proceedings.mlr.press/v336/chen26e.html), abstract | Existing exact divergence characterization; do not propose another unspecified schedule bound. |

These checks do not mean diffusion inference has no open problems. They mean
these formulations are not an adequate novelty claim for our project.

## Narrower question selected

Does a verification computation remain independent of the supposedly hidden
candidate when intermediate representations are shared with a drafting stream
that still sees that candidate? This is a causal property of the computation,
not a claim about whether correct answers generally have high probability.

The selected mathematical check distinguishes per-layer algebraic equality from
end-to-end independence. It does not question the correctness of the diagonal
softmax identity. A useful research contribution would need a practically
important failure in trained decoders and a correction with a competitive cost,
not merely another reminder that attention can propagate information.

[XLNet](https://arxiv.org/html/1906.08237v2) is an essential architectural prior:
target-aware prediction and separation of content/query representations are
old ideas. [SimSD](https://arxiv.org/html/2606.02544v1) already uses controlled
attention contexts for diffusion speculative verification. Any proposed fix
must distinguish itself from these and existing shadow-stream decoders.

No GPU model was loaded, no paid experiment resumed, no external message sent,
and no monitor created during this scout. The next diagnostic is local-first.
