# Fallback novelty triage during the local decoder run

This is an explicit record of *rejected broad formulations*, not a shortlist of
green-lit papers. Searches were done on 4 September 2026 while the frozen
OPDLM on-policy run continued unchanged. Primary abstracts/project descriptions
are enough to disqualify generic novelty claims; they are not a full appraisal
of the cited work's experiments or proof correctness.

| Broad formulation considered | Direct primary prior | Decision |
| --- | --- | --- |
| Stabilize video temporal phase/position sensitivity without retraining | [PAS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_PAS_A_Training-Free_Stabilizer_for_Temporal_Encoding_in_Video_LLMs_CVPR_2026_paper.html) introduces phase aggregation for temporal encoding stability | Do not propose generic temporal phase averaging as new. |
| Allocate denser video observations around query-relevant events | [SlowFocus](https://arxiv.org/abs/2602.03589) already combines local high-frequency sampling and global context | Needs a materially different acquisition objective or result, not another broad adaptive-sampling claim. |
| Camera-compensated representations for VideoLLMs | [Geometry-Guided Camera Motion Understanding](https://arxiv.org/abs/2603.13119) and [ACaM](https://1yuwen.github.io/ACaM-Project-Page/) | A generic camera/object separation benchmark or geometry augmentation is insufficient novelty. |
| Agents should avoid duplicate side effects on retries | [IdempotencyBench author repository](https://github.com/gssanjana4/idempotencybench) explicitly builds this assay; [ReliabilityBench](https://arxiv.org/abs/2601.06112) studies controlled API failures | Basic failure injection plus receipts/idempotency keys is already a known direction. Repository claims are not independently reproduced here. |
| Explicitly revoke superseded memories instead of appending them | [TEPA](https://arxiv.org/abs/2608.07429) provides lifecycle validity/revocation and tests drift | Do not reopen the old dialogue-memory idea under this generic title. |
| Reweight GRPO completions by redundancy/diversity | [MMR-GRPO](https://arxiv.org/abs/2601.09085) explicitly introduces diversity-aware reward reweighting | Generic duplicate-discounted rewards are occupied. |

These collisions do not imply there are no opportunities in video, memory or
RL. They prevent spending on experiments whose headline already matches a
released method. No dataset/model was acquired, code implemented or experiment
launched for these formulations during this triage.

The live cache experiment has its own direct prior collision: section 2.3,
footnote 2 of [the confidence-remasking re-evaluation](https://arxiv.org/html/2606.12232v1)
already notes multi-layer indirect leakage, and section 3.2 tests independent
checking. Our paper burden is therefore a useful, defensible advance, not the
existence of leakage. Finish the existing experiment and review practical gains
and token coverage before spending on any expansion.
