# Fork-suppression follow-up feasibility

Revisited the original CLARA audit and shortlist. CLARA's general mathematical
claims remain rejected; OSH remains excluded from this task. The shortlist's
thinking-model distillation question has not been directly tested by our
nonthinking MATH policy-value bank.

The [July paper](https://arxiv.org/html/2607.05184v1) reports degradation under
privileged-context distillation and explicitly leaves causal attribution open.
It already includes sparse/dense context, nonprivileged distillation, budget
curves and marker analyses. Main training uses 15k examples with 4096-token
rollouts; evaluation reaches 38912 tokens. A small new run cannot be called a
faithful replication of that study.

## Distinct candidate question, not an admitted experiment

Does restoring useful reconsideration behavior recover lost accuracy after
privileged distillation, beyond simply lengthening responses? A convincing study
needs pre/post training checkpoints, a demonstrable accuracy loss, and matched
interventions. Compare targeted restoration with random-position intervention,
equal extra token budget, ordinary prompting, and no intervention. Evaluate
correctness with an executable verifier and cluster by problem. Marker frequency
alone cannot establish restored reasoning. Effects should replicate on independent
problems before training a remedy. This is a design question, not a novelty claim.

## Checkpoint discovery

Saved Hugging Face API responses in `artifacts/fork_distillation_source_20260910`.
Paper-ID and fork-suppression searches returned no models. Broad OPSD search was
limited to100 entries, so it cannot establish absence. Two plausible records
were inspected: `u-opsd/qwen3-4b-thinking` identifies a different paper
(2601.18734); `zen-E/qwen3-8b-think-math-step100-opsd` has no model card metadata.
Neither is authenticated as this paper's trained checkpoint. No weights fetched.

Next prerequisite is an authenticated pre/post pair or a independently specified
training study with measured compute. Do not silently substitute an unrelated
OPSD adapter. No neural launch, paper qualification, or claim that all model
artifacts are unavailable follows from this scoped search.
