# Memory Grafting on Pythia: Step 2 benchmark

## Status and scope

**Complete engineering benchmark; not a scientific experiment.** The run used
seed `26091102` and config SHA256
`8a6c236a50c4fcc184e024cdcfb885d6bcd9a27fd36598fdbc64d48de2be8085`.
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
- Engram fallback: four hash heads per n-gram order, approximately 16,384 rows
  per head and 32 dimensions per head.
- Recipient graft inserted before GPT-NeoX layer 4.
- Full backbone and graft trained together with AdamW at learning rate `5e-5`
  and weight decay `0.01`.
- Sequence length 256, micro-batch 16, 4,096 tokens per optimizer step, 128
  optimizer steps, and 524,288 total training tokens.

The first loss was 4.7122, the final loss was 4.3912, and the mean of the final
eight losses was 4.5043. These confirm finite optimization only; they are not a
registered capability or quality measurement.

## Measured performance

| Measurement | Result |
|---|---:|
| Full 128-step training wall time | 4.7779 s |
| Full-run throughput | 109,731 tokens/s |
| Post-warmup throughput | 114,706 tokens/s |
| Peak CUDA memory allocated | 5.00 GiB |
| Peak CUDA memory reserved | 5.98 GiB |
| Offline 3,072-row bank construction | 0.4752 s |
| Corpus tokenization | 2.3929 s |
| N-gram counting and selection | 5.1252 s |
| Cold dataset load/download | 4.4064 s |
| Model load | 0.5860 s |
| Graft attachment and compression setup | 0.2925 s |
| Total end-to-end wall time | 19.8144 s |

Training consumed 0.00133 GPU-hours; the complete process occupied the instance
for 0.00550 hours. At this measured 160M configuration, accelerator memory rather
than the remaining approximately 49.94 GPU-hour budget is not a near-term
constraint. Runtime for larger backbones, larger tables, additional seeds, or
longer sequences has not been measured and must not be inferred from the 160M
number as if it transferred unchanged.

## Parameters and evidence

| Component | Count |
|---|---:|
| Pretrained backbone | 162,322,944 |
| Trainable hash table | 6,316,736 |
| Other trainable graft parameters | 1,775,616 |
| Total trainable graft | 8,092,352 |
| Combined trainable parameters | 170,415,296 |
| Frozen exact-bank values | 2,359,296 |

The raw report `artifacts/pythia_memory_graft_step2.json` has SHA256
`fbb023bad69c7ce8fb5f79f2c3b62d2ecea91321544b80f544450b61aa6915f4`.
The execution code was committed at
`fd4c624719fc74794d2d6fa1da88544ef1b84b04`. In accordance with the binding
verification rule for developmental engineering checks, the recorded seed and
config hash are the verification record; no sealed manifest or full replay was
created.

## Boundary

Step 2 establishes measured feasibility only. It does not validate language
quality, conditional-memory utilization, poison learnability, row localization,
or ablation specificity. Step 3 must derive each apparatus gate from the S1
ablation estimand and justify model, table, bank, token, and seed scale against
the remaining budget before anything is frozen. No Step 3 preregistration has
been written or frozen.
