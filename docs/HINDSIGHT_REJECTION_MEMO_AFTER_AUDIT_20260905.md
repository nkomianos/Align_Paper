# Skeptical ICLR rejection memo

5 September 2026. Red-team review of the proposed
[Hindsight pitch](HINDSIGHT_PAPER_PITCH_AFTER_AUDIT_20260905.md), using the audited
evidence available today. Recommendation on that evidence: reject.

The central objection is simple: the investigators engineered worlds that cannot
be distinguished from immediate feedback, then supplied delayed labels that
distinguish them by assumption. The proposed correction is standard difference
estimation. Without a defended measurement instrument and a useful result against
ordinary learners given those labels, this is supervised learning with better
labels plus a small nonidentification example. Neither piece currently establishes
the claimed research contribution.

The theorem's scope is narrower than the original story. Its `.2` fixed-action
regret is not a lower bound for every deterministic data-dependent learner; a
deterministic rule using random historical data attains worst expected regret
`.16`. The corrected universal expected-risk bound is `2/15`. Moreover, knowing
the deployment user's baseline state removes regret in the witness. A reviewer
would reasonably ask why the restricted population decision is the relevant
personalization setting and why matching the post-response state is the desired
objective. The paper must state those choices rather than imply an unavoidable
welfare failure for all adaptive assistants.

The empirical endpoints currently do not answer the theorem. G1 measures
distance to a trained oracle; moving farther from an imperfect oracle can improve
correctness. EndoPAHF predicts fixed delayed labels; its evaluated action does
not induce the target transition. All DEV users appear in learning. Option
rotations and anchor panels are dependent observations, and the old power
exercises do not simulate the whole proposed decision rule. These defects are
correctable as experimental design, but repairing them creates a future assay,
not retrospective empirical support.

The strongest simple baseline is already damaging. In the clean robustness grid,
ordinary anchor regret `.00380385` is below augmented-IPW `.00417692`. Published
baselines must also receive the sparse delayed information available to the new
method. Otherwise a win measures access to labels. Ensemble counts, updates,
tokens and compute need separate accounting. A gradient alignment improvement
without a decision benefit would not resolve this concern.

The novelty margin is small. SLIFT covers feedback decomposition; PUMA covers
inferred user-state dynamics; Privileged Likelihood directly studies why
likelihood-based privileged gradients need not improve utility. Prior human
expression-versus-reward intervention work further narrows the space. The
restricted witness and standard estimator do not justify a broad causal-learning
or generic measurement-validity claim. The exact comparisons must be explicit,
including what each prior observes, intervenes on and evaluates.

What would change this recommendation? First, a prospectively qualified utility
result against pooled anchor SFT, anchor SDPO, a fixed mixture and simple readout
controls, all given the same labels; then replication, matched compute and fair
published baselines. Second, an independently defensible persistence measurement
and an honest connection between its estimand and the theorem. Third, a complete
manuscript whose claims are reproducible from sealed evidence. The reduced
six-arm screen resolves only the first, preliminary method question. It does not
validate a human instrument or remove the estimand mismatch. If no credible
measurement route exists by September 8, or no replicated package and full draft
exists by September 11–12, a later venue is the scientifically responsible plan.
