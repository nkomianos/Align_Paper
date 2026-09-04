# Recommendation: one natural-data single-profile reproduction, conditional go

Planning only, before reading the new frozen-forward diagnostic. No implementation or launch. This is a known-method replication prerequisite, not a new paper experiment.

## Why it is worth considering, and what it cannot establish

Our multi-user formatting pilot simultaneously departed from released online personalization in user allocation, objective variant, LR and adapter capacity. It also failed preservation of already-correct behavior under generic approval. Repeating that bespoke assay with another parser or format message would provide little scientific value.

A single-profile run on the released natural prompt distribution would answer a useful narrower question: **can we establish beneficial interaction learning with a substantially closer released recipe before testing any new causal claim?** It would not identify which old departure caused collapse. Do not describe a bundled default-recipe change as isolating the effect of LR, loss or user routing.

## Smallest defensible configuration

Use pinned upstream `3b17d2a67bd2565b9fbda495fd16a485406aa954`, original TL;DR preprocessing and the original `concise_casual_beginner` profile. Use the cached native Qwen3-4B policy and a separately pinned Qwen3-8B user simulator/judge, matching the paper's simple stylistic setting. Do not create opaque user IDs, conflicting profile streams, or new format grammars.

Use released `full_distillation` (student top20 plus tail, reverse KL), AdamW5e-6, epsilon1e-6, gradient clipping1, zero weight decay, LoRA256/512 over attention and MLP projections, one update per episode. Keep the upstream hindsight message and simulator persona. Any native-ID bridge, no-vLLM execution or Transformers5.6 adaptation must be expressly documented; it is method-level replication, not bitwise replication of the published environment.

Freeze a small natural-data split before any outputs:16 calibration prompts,64 adaptation prompts and32 evaluation prompts, deduplicated at the source-post level, with no prompt rewriting from observed model mistakes. Ordinary prompts contain no desired style. Do not select calibration examples to force a favorable mix of successes and failures.

## Qualification and fixed stopping rules

Before training, compare ordinary responses, explicit-style responses and hindsight-conditioned responses on all16 calibration prompts. A blinded, order-balanced style judge must distinguish the explicit-style condition from the original where headroom exists. Manually inspect all16 short feedback/response tuples for content corruption, merely copied praise and judge incoherence. Separate the initially satisfactory and unsatisfactory strata using the frozen rubric; evaluate preservation and recovery in both. The original failure was not merely lack of recovery.

Preregister conservative engineering thresholds before generation: no content-corruption cases; at least80% preservation among originally satisfactory cases and at least75% recovery among initially unsatisfactory cases; at least four examples in each stratum. If those strata are unavailable or judging is inconsistent, stop as unqualified/insufficient headroom without changing the profile, examples or rules. Small denominators make this an apparatus check, not a statistical claim. The same simulator acting as judge is still not external validation; order reversal and human inspection are essential controls.

If qualified, generate and save no-adaptation evaluation responses, perform exactly64 updates, then evaluate the fixed final adapter on the same32 prompts. Preserve every update and checkpoint; no interim checkpoint selection or extra updates triggered by weak results. Primary descriptive outcome: blinded order-balanced preference win/tie/loss versus baseline, alongside summary factual consistency and length. A promising control needs clearly positive net preference wins with no factual regression, not simply shorter outputs. Report uncertainty and raw examples, and do not treat a marginal sign as a go signal for the original paper.

An unqualified or non-improving run ends this reproduction attempt. It does not refute SDPO:64 interactions remain a small-budget replication and the environment/simulator differ in ways requiring disclosure. No endogenous-feedback comparison follows automatically.

## Runtime and readiness

The current model/runtime can plausibly support the arithmetic on one GH200; the upstream shell launcher cannot be used unchanged because it allocates multiple GPU indices. A second8B model, pinned dataset split, PEFT support and compatible attention backend must first be checked without altering the verified live environment. Missing dependencies or downloads are a staging blocker, not a scientific failure.

With staged assets, budget roughly1–2 GPU-hours for this fixed small study, depending on actual response/judge lengths; a sub-hour promise is not supported yet. Measure complete end-to-end latency on the frozen calibration calls, including both feedback and judge, and project the remaining fixed call count. Do not substitute a smaller simulator or truncate long responses silently to satisfy the budget. If projection exceeds two hours, do not start training. Reserve retrieval time separately.

## PI recommendation

Conditional go to **prepare** this closer reproduction, only if the frozen-forward diagnostic supports a concrete teacher/estimator explanation and we still want to pursue the original interaction-learning direction. No-go to immediately launch another synthetic formatting run or a four-way hyperparameter sweep. The main research question remains causal identification of expression versus persistent transition; even a successful replication would provide only a competent experimental learner, not that contribution.
