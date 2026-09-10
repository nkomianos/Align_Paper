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

## Official release located in follow-up

Found [the official implementation](https://github.com/princeton-pli/rethinking-opsd-for-thinking-models)
and pinned commit `46ccfbddeb92c3bb8b9602a1e6644eb9fa8f7c04`. Complete tree and
commit metadata plus README saved alongside earlier search receipts. README
SHA256: `8fe815b16f727283ea960fcf4012a2fabba2fcbec2c928ef54b268e4dfac1454`.
It specifies a locked training environment and reports 5–12 hours on eight H100s
per run, plus separate evaluation costs. It gives local output checkpoint paths,
not downloadable trained weights. GitHub releases and the lead author's
`kaur-sim` model listing returned empty arrays. This is stronger source evidence
than keyword search, but still not proof of universal checkpoint unavailability.

Decision: do not queue the full reproduction within this session. Its reported
40–96 H100 GPU-hours for one dense training run are not directly convertible to
our hardware budget without benchmarking, and exclude replication/evaluation.
A reduced-data pilot would be a new developmental study with uncertain formation
of the reported degradation, not an economical substitute for verified checkpoints.
No upstream training script executed and no author contacted.

## Expanded remedy literature changes the novelty assessment

The initial memo did not cover two directly relevant remedies. Read the primary
abstracts and available method sections, rather than relying on search summaries:

- [Purified OPSD](https://arxiv.org/html/2607.02234v1) constructs a teacher target
  using a reference-only contrast, a clean base distribution and a bounded PMI
  correction. It already studies preserving reflective behavior during
  self-distillation. Its reported setup uses rank-64 LoRA, 1,024-token training
  completions and up to 200 steps, unlike the full reproduction described above.
  Its reported best-checkpoint benchmark results need their own selection audit;
  we have not independently validated them.
- [Diagnosing and Mitigating Thinking Collapse](https://arxiv.org/html/2607.10805v1)
  explicitly studies entropy-based gradient masking and suppressive updates at
  uncertain decision forks. Protecting those tokens is therefore already an
  investigated intervention, not a fresh proposal here.
- [One Symptom, Three Levers](https://arxiv.org/html/2608.25936v1) organizes this
  literature by token weighting, privileged information and teacher dynamics.
  It explicitly reports no new experiments and is a discovery map, not evidence
  independently confirming the empirical papers.

This changes the next-action rule: any new prevention method needs comparisons
to these approaches, not merely vanilla OPSD. Post-training causal restoration
may still be distinct, but marker preservation or entropy gating alone does not
establish novelty. These papers do not authenticate a pre/post checkpoint pair
for our original restoration experiment. A scoped Hugging Face search for
`purified opsd` returned no matches; it is not exhaustive.

The adapter setup makes a smaller independent distillation study more plausible
than the eight-H100 reproduction, but no absolute runtime was measured here.
Do not extrapolate a reported relative overhead into an estimate for our AWS
host. Before implementing another trainer, specify the remaining causal question,
its missing baseline and a formation test that establishes the behavior being
repaired. No new model, dataset, training or inference run occurred in this screen.

Subsequent exact null-reference analysis adds another necessary baseline:
`PMI_NULL_REFERENCE_CONTROL_20260910.md`. The purified target can sharpen the
question-only distribution even when the reference contributes no information.
Compare against that question-only density-ratio effect before attributing a
future empirical gain to reference purification. No neural comparison has run.
