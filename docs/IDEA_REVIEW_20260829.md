# ICLR 2027 Idea Review — 2026-08-29

## Outcome

Two candidates survived the current primary-source screen and received frozen
small gates:

1. **Environment-executed effect consistency** — primary recommendation.
2. **VLM patch-phase instability** — independent backup.

The semantic-ancestry RAG formulation remains killed. Its positive
cross-ancestor effect is an archived clue, not authorization to tune the failed
selector or weaken the specificity test.

## Recent open questions screened

| Direction | PI decision | Reason |
| --- | --- | --- |
| Environment effects as tool-agent UQ | Freeze G0 | 2026 trajectory UQ finds consistency promising but still compares action structure or uses an LLM equivalence judge; exact same-state execution provides a distinct causal measurement and a router consequence. |
| Patch phase as a cause of MLLM primitive failure | Freeze backup G0 | KidVis/BabyVision establish the gap but not this mechanism. ViT shift sensitivity is old, so continuation requires encoder-period locking and compute-matched mitigation in two MLLMs. |
| Clarification from uncertainty/EVPI | Cull | Information-gain clarification, self-gating, and structured EVPI clarification already occupy the natural method space: [IG clarification](https://arxiv.org/abs/2606.03135), [self-gated clarification](https://arxiv.org/abs/2606.11349), [structured uncertainty/EVPI](https://arxiv.org/abs/2511.08798). |
| Action-preserving observation compression | Cull | [AGORA](https://arxiv.org/abs/2605.26596) and [CoACT](https://arxiv.org/abs/2607.02911) directly learn action-grounded/action-preserving compression; another summary policy would be incremental. |
| Causal memory retrieval and repair | Cull | The 2026 memory survey identifies causal retrieval and consolidation as open, but [causal episodic repair](https://arxiv.org/abs/2608.05906), [ActMem](https://arxiv.org/abs/2603.00026), and [MemReranker](https://arxiv.org/abs/2605.06132) already occupy the immediately testable formulations. |
| Byte masked-diffusion locality | Cull | [The Efficiency Gap in Byte Modeling](https://arxiv.org/abs/2605.12928) identifies broken local contiguity, but blockwise locality and schedule fixes already appeared in [Jigsaw/Scatter](https://arxiv.org/abs/2604.24832) and [locality-aware time sampling](https://arxiv.org/abs/2605.13026). |
| Tokenization-equivalence decoding | Cull | The anomaly is real, including 2026 phantom edits in [Say Anything but This](https://arxiv.org/abs/2601.14658), but marginalizing multiple tokenizations, canonical generation, and multilingual retokenization training already cover the obvious remedies: [marginalization](https://arxiv.org/abs/2306.17757), [non-canonical robustness](https://arxiv.org/abs/2607.26831). |
| Adaptive reasoning depth / early rejection | Cull | [Think Deep, Not Just Long](https://arxiv.org/abs/2602.13517), [AdaPonderLM](https://arxiv.org/abs/2603.01914), and multiple 2026 stopping policies occupy the mechanism and mitigation. |

## Ranking

Effect consistency is the better ICLR bet because it is directly downstream of
a fresh empirical gap, has exact non-LLM supervision, creates a deployable
uncertainty/routing primitive, and is cheap to falsify. Patch phase is riskier:
the architectural failure is known, and reviewers may call a synthetic-only
result an application of shift-equivariance work. It is retained because the
cross-encoder period prediction is unusually clean and a positive result would
explain a conspicuous 2026 benchmark anomaly.

Neither candidate is described as “high confidence acceptance” before data.
The honest high-confidence decision is only that these are now efficient,
auditable bets with hard kill criteria.
