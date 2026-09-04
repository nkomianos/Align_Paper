# SDPO release defaults and one prospective diagnostic

Research plan only. No implementation, neural forwards, training or rerun authorized by this document. Frozen v3 remains unchanged.

## Release comparison, checked directly

Pinned source: `lasgroup/user_interactions` commit `3b17d2a67bd2565b9fbda495fd16a485406aa954`.

| Setting | Released online configuration/launcher | Our v3 pilot |
|---|---|---|
| Objective default | `full_distillation`: reverse KL on student top20 vocabulary entries plus tail | Released `simple_signal`: one sampled token per prefix, detached log-ratio advantage |
| AdamW learning rate | 5e-6 | 1e-4, twenty times larger |
| Advantage clipping | signal_clip=0, disabled | Disabled; matches this default |
| Gradient norm clipping | 1 | 1 |
| Epsilon / weight decay | 1e-6 / 0 | Same |
| LoRA | Launcher enables it; rank256, alpha512, attention AND gate/up/down projections | rank16, alpha32, attention only |
| Personalization | One profile per online stream; additional profiles can be evaluation judges | Eight conflicting profiles in one ID-conditioned adapter |

The dataclass defaults `use_lora=False`, but the released shell launcher explicitly enables it. Both statements must be distinguished. The released objective named full_distillation is a top20-plus-tail approximation, not literal full-vocabulary KL. Its top-k entries use the student's support. [Configuration](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/online_sdpo_updater_config.py), [loss implementations](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/online_sdpo_updater.py), [launcher](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/scripts/eval_online_sdpo.sh)

Our pilot is source-faithful to a released loss VARIANT, with a documented native-ID bridge. It is not the default online recipe. The larger LR, smaller restricted adapter, different user allocation and sampled objective could exacerbate collapse; the existing single run does not identify which causes dominate. Generic approval inserted into an LLM prompt is not the Bayesian posterior conditioned on an actual observation of approval. It can change the response distribution in an unrelated direction. This is a known privileged-information/teacher-validity concern, not a newly established paper contribution.

## ONE bounded frozen-forward diagnostic

Question: **Does truthful approval already redirect a competent initial model away from its correct response, and does later saturation suppress sampled-token correction despite a corrective teacher?**

Use initial, step16 and final saved adapters; steps4/5 adapters do not exist, so do not pretend to reconstruct their exact states. Fix all eight initially correct calibration cases plus eight initial mismatches selected by lexical case ID before new neural evaluation. This is post-hoc mechanism development, not independent validation.

At each checkpoint evaluate the SAME original completion tokens and their SAME prefixes, with four contexts: base; empty hindsight block; actual frozen feedback; explicitly stated original preference via the existing calibration prompt. No sampling, no parameter changes. For each of the192 teacher-forced forwards, retain full vocabulary logits only at the first response position and first punctuation/layout decision, plus response-token log probabilities. Match all other tokenization and settings to archived code. Confirm saved initial base probabilities where available.

Offline from those logits, calculate the exact vocabulary expectation and variance of the one-token stopped-advantage estimator, and the released top20-plus-tail objective's logit gradient. The score-function expectation is sum_y p(y)[log q(y)-log p(y)](onehot(y)-p); use frozen p and q. These are local logit-space diagnostics, NOT parameter-gradient or optimizer replay. Include the saved sampled token and whether it was representative of the distributional gradient.

Interpretation: approval-induced loss of target likelihood at the initial checkpoint supports teacher-conditioning failure independent of training. Deterioration only after adaptation supports model-change dependence. Difference from the empty-block control helps isolate generic prompt perturbation. Weak local sampled gradients under near-deterministic p, compared with early values, supports saturation; expectation/variance distinguishes one realization from its expectation. None of these interventions uniquely attributes changes to LR rather than optimizer momentum or user interference. That would require a separate randomized training experiment, not more interpretation of one trajectory.

Budget estimate: model load plus192 short-prefix forwards, likely10–25 minutes given the recorded short-context run speed; measure first12 forwards and stop if the fixed45-minute ceiling is at risk. No new dataset or model download. This is the recommended next diagnostic if approved, rather than another64-update format repair.

## Exact upstream online reproduction within one hour?

**Not presently a credible one-hour commitment.** The literal shell launcher requests separate vLLM inference/training devices and starts the user/judge at GPU2; it does not fit one GH200 unchanged. The underlying Python evaluator can avoid vLLM, but defaults to flash_attention_2. Its pinned stack is torch2.7.0, transformers4.57.6, accelerate1.6.0, peft0.15.1, datasets3.5.0, trl0.24.0, plus optional vLLM/bitsandbytes and simulator/judge dependencies. The current verified environment is not that exact dependency stack. Compilation/downgrading the live environment is not acceptable.

A faithful *single-GPU adaptation* would use an isolated environment and the pinned Python evaluator: Qwen3-4B policy, one `concise_casual_beginner` profile, native TL;DR prompts, Qwen3-8B simulator and local judge, released full_distillation/top20+tail, lr5e-6, LoRA256/512 over all released projections, one update per episode, no vLLM. Freeze15 training prompts and32 held-out prompts before generating anything; this is a small reproduction smoke test, not the paper's250/500-interaction studies. Acquisition of the pinned8B model and datasets, flash-attention compatibility, and representative output length have not been qualified, so do not quote a guaranteed runtime.

Required controls: baseline and final responses to identical held-out prompts; explicit-style oracle; recovery AND preservation under feedback; style-judge order reversal/tie handling; content/factual-quality checks; independent feedback/judge audit because shared simulator/judge agreement is not external validity. Retain native responses, checkpoints and all updates without choosing a winning intermediate checkpoint. Shortening all outputs or replacing natural prompts/users with our format simulator would be another apparatus, not an exact reproduction.

Decision: prefer the small no-update diagnostic above. Prepare any upstream reproduction offline with downloads and environment validation before renting another hour, and label it reproduction rather than a new paper gate.
