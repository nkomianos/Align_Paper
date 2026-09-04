# SDPO v3 shared-user scope audit

Date: 2026-09-04. Interpretation audit while the frozen v3 run is active. No source, data, thresholds, optimizer or schedule changes. No new run proposed for automatic execution.

## Finding

**The current apparatus is harder in a materially different way than the published online personalization experiment.** One shared adapter must acquire eight arbitrary opaque-ID-to-format associations from eight updates per user, interleaving four mutually exclusive strict formats. The original online setting does not require such routing among conflicting users. Consequently, failure of the fixed 64-update pilot is not evidence that SDPO cannot personalize, cannot learn from truthful corrections, or that the original causal paper hypothesis is false.

The data are still a coherent task: IDs recur across held-out contents and define stationary preferences. But learning these arbitrary associations is an additional prerequisite, not something established by correctly executing an explicitly supplied format instruction.

## Direct evidence

The locally inspected upstream checkout is pinned to `3b17d2a67bd2565b9fbda495fd16a485406aa954`:

- `eval_online_sdpo.py` initializes one training updater (line338), creates its user simulator with `style=args.style` (lines372/387/391), and trains that updater on successive prompts. Additional styles construct secondary judges; they do not form an interleaved user-ID-conditioned training population.
- `scripts/eval_online_sdpo.sh` takes one `STYLE` (line47), defaults `LOSS_MODE` to `full_distillation` (line50), and explicitly passes `--use_lora` (line107).
- `online_sdpo_updater.py` describes its core setting as live single-user chat training.

[Pinned evaluation source](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/eval_online_sdpo.py), [pinned launcher](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/scripts/eval_online_sdpo.sh)

The paper's personalization section trains toward one style profile. Its changing-preference study flips the preference after250 interactions; its other continual experiment introduces three complementary preferences in500-interaction blocks. These are not demonstrations of learning eight unrelated identity-conditioned preference bindings from eight examples each. The paper also has a separate large offline interaction-learning setting; this audit does not assert that all SDPO usage is single-user. [Primary paper, section4.2](https://arxiv.org/html/2603.12273v1)

## What our calibration does and does not establish

Explicit-format success and hindsight recovery show that the model can express the desired format and interpret the supplied correction. They do not show that its shared low-rank adapter can efficiently memorize identity-format bindings while retaining competing bindings under this schedule. A globally reinforced default format could improve some users and harm others without demonstrating a flaw in the SDPO signal.

No historical user feedback is included in an ordinary new-episode prompt. Thus the binding must be stored in parameters. This is not an accidental information leak; it is an extra associative-learning requirement. The no-adaptation comparison alone cannot isolate whether an unsuccessful result reflects weak signal, update budget, interference, or limited identity routing.

## Is existing documentation sufficient?

It already discloses eight opaque users, recurring identities, strict formats, LoRA settings,64 updates and lack of unseen-user claims. It does **not** adequately emphasize that this differs from upstream single-profile personalization. This addendum supplies that interpretation caveat without retroactively moving the frozen positive-control criterion.

One technical clarification: the prior protocol's statement that the released *configuration* defaults to no LoRA must not be paraphrased as saying the released online *launcher* uses full-weight training. That launcher explicitly enables LoRA. Likewise, extracting the upstream `simple_signal` loss establishes fidelity to that released variant, not equivalence to the launcher's default `full_distillation` recipe or every paper experiment.

## Result language and PI action

- A verified positive result supports this particular multi-user learning apparatus; it does not establish endogenous preference shaping or a novel correction.
- A verified failure means **this fixed shared-adapter, sparse identity-conditioned positive control did not qualify**. Preserve it and do not run endogenous-feedback arms on it.
- Do not relabel a failure as a rejection of SDPO, nor silently treat the frozen outcome as an invalid computational run. Its observations remain valid within its narrower scope.
- A fair future reproduction would separately establish single-profile adaptation using a paper-matched loss/budget, then test multi-user routing against an independent trainability baseline. That is a different prospectively designed study, not an authorized extension or repair of this run.
