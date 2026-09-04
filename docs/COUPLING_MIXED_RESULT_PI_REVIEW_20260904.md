# Mixed coupling results: PI review and optional clarification

This review incorporates the completed laptop result after the earlier larger-
model GPU result. It preserves both primary reports and the earlier negative
practical result; it does not retroactively change their tests or thresholds.

## Evidence, not a blanket failure claim

The small-model pair has hierarchical F1 variance ratios of 0.660, 0.631 and
0.714 versus independent, token-clock and byte-clock sampling. The larger pair
has ratios 1.058, 1.252 and 0.947. The small-pair observation is genuinely
promising as a development observation and should not be buried. Neither run
alone establishes general utility or model-size-dependent behavior. Model pair
and execution platform changed together (laptop CPU versus GPU); this is not a
controlled scaling experiment.

For the small pair, the observed F1 variance reduction versus independent is
0.03430: sampled marginal-variance differences contribute 0.01183, and twice the
covariance improvement contributes 0.02247. Against byte-clock, marginal changes
contribute 0.01571 of the total 0.02675. Thus the benefit is not merely a marginal
sampling artifact, but those fluctuations are too large to ignore. Covariance
improvement is the appropriate mechanism target: true marginal variances are
the same under every coupling policy.

## Explicitly post-hoc uncertainty sensitivity

`scripts/coupling_posthoc_seed_uncertainty.py` preserves model/policy pairing
within each question and resamples seeds jointly. It also performs a nested
question-and-seed bootstrap. It checks all evidence hashes and agreement of
recomputed primary moments before analysis. These are descriptive sensitivity
checks, not new registered significance tests; eight question clusters and
sixteen seeds cannot support precise generalization claims.

Small-pair nested 2.5th–97.5th percentiles for the F1 variance ratio:

| Baseline | Interval |
| --- | --- |
| Independent | 0.363–1.114 |
| Token clock | 0.346–1.020 |
| Byte clock | 0.388–1.173 |

All corresponding covariance-improvement intervals include zero. The larger-
pair intervals are wider and likewise do not establish a practical gain. Seed-
only resampling favors the small pair versus independent/token-clock, but not
decisively versus byte-clock. No bootstrap variant should be selected because
its interval happens to exclude one.

Outputs are separate:
`artifacts/squad_coupling_dev_v1_posthoc_seed_uncertainty.json` and
`retrieved/gpu_coupling_20260904T0840Z/posthoc_seed_uncertainty.json`.

## Conditional marginal proof audit

In ideal independent random fields, the proof is sound. The group categorical
probability is its summed token mass; conditional selection inside that group
has the correct normalized token probabilities. Separate group/token streams
give independence. Each model's conditioning depends only on its own history.
The clock `(byte_offset << 16) | consecutive_silent_events` is strictly advancing:
positive-length tokens increase the offset and reset the silent count; silent
tokens increase only that count. The bounded token cap prevents overflow.
Consequently, a model does not revisit a used clock, even though the next clock
depends on previous sampled tokens. Cross-model sharing does not itself alter
either marginal. Special and padded output rows remain in support.

This is an ideal-arithmetic argument, not a literal proof that finite 64-bit
hash/mixer outputs are mutually independent real Gumbels. Hash collisions and
floating point ties have finite probability; the code already labels the stream
pseudorandom. No concrete clock reuse or support-changing bug was identified.

## Decision and prospective constraints

**Paper expansion remains unapproved.** The original protocol says a weak/mixed
result parks the heuristic; therefore another run needs a visible post-DEV
amendment, not a claim that the old gate passed. Nonetheless, one inexpensive
clarifying replication is scientifically defensible given the retained small-
pair signal, provided it does not displace higher-priority qualified experiments.

If the root PI elects to do it, require all of the following before any outputs:

1. Same four frozen model revisions, all run on the same GPU implementation,
   same float32/temperature/token cap and unchanged policies/scorer.
2. Both original pairs, never only the pair that looked positive. Report each
   separately; do not hide the negative pair behind an overall mean.
3. Select 32 new article-disjoint questions by a frozen hash from the remaining
   eligible SQuAD development articles, excluding the eight inspected questions.
   This is fresh within the experiment, not uncontaminated external evaluation.
4. Use 32 new seeds shared across all arms; no stopping after attractive batches.
   The full matrix is 32 × 32 × 4 policies × 4 models = 16,384 completions.
5. Freeze covariance improvement and variance-times-cost against BOTH simple
   coupling baselines as primary practical checks; keep independent as reference.
   Report bootstrap uncertainty, each marginal variance, failures and all arms.
6. No new grouping, prompt tuning, model substitution, semantic parser, cherry-
   picked subgroup or repeated fresh split after seeing the answer.

A positive effect confined to the small pair would establish bounded
heterogeneity, not a broadly useful method or high-confidence ICLR paper. It
would still need a predictive account of when coupling helps and comparison
with complete-byte coupling. A weak result again ends this heuristic's current
experimental sequence. This review itself authorizes no launch or GPU expansion.
