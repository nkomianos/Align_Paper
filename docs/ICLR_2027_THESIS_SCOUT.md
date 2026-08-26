# ICLR 2027 Thesis Scout

## Decision standard

This log records *negative* literature screens as well as candidates.  A GPU
experiment is authorized only after all five conditions below are met:

1. The central claim is distinguishable from the closest 2025--2026 work.
2. The first experiment tests a necessary causal prediction, not merely a
   behavioral correlation.
3. A positive result has a credible mitigation or design consequence.
4. The claim can be tested across at least two independent model families or
   other genuinely independent settings.
5. The proposed paper can be completed and audited before the ICLR 2027
   deadlines (abstract: 18 September 2026 AOE; paper: 25 September 2026 AOE).

Passing a cheap gate is not evidence of publishability.  A candidate becomes a
paper only after it has both a large enough causal effect and an independent
replication/transfer result.

## Closed directions

| Direction | Why it was screened out | Nearest work |
| --- | --- | --- |
| Recurrent-state safety carrier in Qwen3.5 hybrid attention | The pre-registered cache-state intervention found a small recurrent effect (0.168 logits; continuation threshold 0.50) and a dominant attention K/V effect (14.449 logits).  It is not a sufficient mechanism or mitigation. | Local verified result: `docs/HYBRID_MEMORY_G0_RESULT.md` |
| Provenance / self-authorship authority | The forced-likelihood gate found a 0.0277 pp self-minus-external effect against a 10 pp continuation threshold. | Local evidence: `retrieved/provenance_authority_g0_20260825/report.json` |
| Multilingual mechanistic safety steering | Recent work already supplies multilingual consistency training, sparse weight editing, and English-derived steering across languages.  A simple cross-lingual safety-subspace study would be derivative. | [Align Once, Benefit Multilingually](https://openreview.net/pdf?id=ueknOG1wXL); [Sparse Weight Editing](https://arxiv.org/abs/2602.22554); [BabelSteering](https://arxiv.org/abs/2608.16577) |
| Hybrid recurrent-state poisoning | The attack surface and an explicit hybrid-LLM defense already exist.  Reproducing or lightly extending either would not meet the differentiation standard. | [HiSPA](https://arxiv.org/abs/2601.01972); [CLASP](https://arxiv.org/abs/2603.12206) |
| Safety effects of KV-cache quantization | A contemporaneous eleven-model study already gives a geometric mechanism, diagnostic, and recovery method. | [Alignment Collapse Under KV Cache Quantization](https://arxiv.org/abs/2606.09864) |
| Sparse-MoE routing as a safety mechanism | Unsafe routes, routing-aware jailbreaks, inference-time expert steering, and a router-preserving safe-fine-tuning method already cover this causal object and its natural mitigations. | [Sparse Models, Sparse Safety](https://arxiv.org/abs/2602.08621); [RouteHijack](https://arxiv.org/abs/2605.02946); [SafeMoE](https://openreview.net/pdf/277ee9ab2217156725062e3eed1aaa22a15ed779.pdf) |
| Safety in diffusion language-model denoising | Multiple papers already cover the initial mechanism, attacks, and denoising interventions. | [Fragile Guardrail](https://arxiv.org/abs/2602.00388); [Re-Mask and Redirect](https://arxiv.org/abs/2604.08557); [Mechanistic Safety Exploits](https://arxiv.org/abs/2608.07430) |
| Tool provenance / authority | This is actively covered by agent-provenance and formal safe-tool-use work. | [ProvenanceGuard](https://arxiv.org/abs/2607.01236); [Verifiably Safe Tool Use](https://arxiv.org/abs/2601.08012) |

## Implication

No direction in this table should be revived by changing a threshold after a
negative result or by repackaging an existing method.  The next candidate must
introduce a new causal object and practical mitigation, then pass the decision
standard above before spending further GH200 time.
