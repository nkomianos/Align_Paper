# Memory Grafting on Pythia: Step 2 benchmark

## Status and scope

**Complete engineering benchmark; not a scientific experiment.** The run used
seed `26091102` and corrected config SHA256
`90c58b8b2c324da32d86773d1b44516768a1d4caff79186e0b050500f620c5a7`.
It introduced no gate or scientific threshold and did not estimate the proposed
poison-localization effect.

An initial invocation stopped before loading data or weights because the isolated
environment lacked the repository-pinned `datasets` package. After installing
`datasets==5.0.1`, the same committed config ran without modification.

## End-to-end workload

- Pinned `EleutherAI/pythia-160m` revision
  `b56d9bee36300031aeea723b73c4d62ac7fa71a2`, bfloat16.
- Pinned `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`,
  `wikitext-103-raw-v1` training split.
- 1,500,000 real-language tokens materialized from 14,666 rows.
- Frozen exact bank: the 1,024 most frequent n-grams at each of orders 2, 3,
  and 4, for 3,072 rows and 2,359,296 frozen value elements. Offline values came
  from donor layer 6.
- Engram fallback: 2- and 3-grams, four hash heads per order, approximately
  16,384 rows per head and 32 dimensions per head.
- Recipient graft inserted before GPT-NeoX layer 4.
- Full backbone and graft trained together with AdamW at learning rate `5e-5`
  and weight decay `0.01`.
- Sequence length 256, micro-batch 16, 4,096 tokens per optimizer step, 128
  optimizer steps, and 524,288 total training tokens.

The first loss was 5.0029, the final loss was 4.7670, and the mean of the final
eight losses was 4.9231. These confirm finite optimization only; they are not a
registered capability or quality measurement.

## Measured performance

| Measurement | Result |
|---|---:|
| Full 128-step training wall time | 4.7443 s |
| Full-run throughput | 110,509 tokens/s |
| Post-warmup throughput | 115,226 tokens/s |
| Peak CUDA memory allocated | 4.99 GiB |
| Peak CUDA memory reserved | 5.97 GiB |
| Offline 3,072-row bank construction | 0.4797 s |
| Corpus tokenization | 2.3675 s |
| N-gram counting and selection | 4.8982 s |
| Cached dataset load | 0.1969 s |
| Model load | 0.5614 s |
| Graft attachment and compression setup | 0.2760 s |
| Total end-to-end wall time | 15.2110 s |

Training consumed 0.00132 GPU-hours; the complete cached process occupied the instance
for 0.00423 hours. At this measured 160M configuration, accelerator memory rather
than the remaining approximately 49.94 GPU-hour budget is not a near-term
constraint. Runtime for larger backbones, larger tables, additional seeds, or
longer sequences has not been measured and must not be inferred from the 160M
number as if it transferred unchanged.

## Parameters and evidence

| Component | Count |
|---|---:|
| Pretrained backbone | 162,322,944 |
| Trainable hash table | 4,206,720 |
| Other trainable graft parameters | 1,579,008 |
| Total trainable graft | 5,785,728 |
| Combined trainable parameters | 168,108,672 |
| Frozen exact-bank values | 2,359,296 |

The corrected raw report
`artifacts/memory_graft_engineering/pythia_memory_graft_step2_faithful.json`
has SHA256
`4251bd168832c505a524e882cdf1f155c119a3ec33cee3356a8cbd051bd825a8`.
The corrected execution code was committed at
`54a7cb2de3795a832f1074bd9f6e6cc055d2516f`. The original benchmark used a
non-paper fallback 4-gram table; it remains preserved and this report supersedes
it. In accordance with the binding
verification rule for developmental engineering checks, the recorded seed and
config hash are the verification record; no sealed manifest or full replay was
created.

## Boundary

Step 2 establishes measured feasibility only. It does not validate language
quality, conditional-memory utilization, poison learnability, row localization,
or ablation specificity. Those requirements are addressed separately in the S1
preregistration and are not inferred from this benchmark.
