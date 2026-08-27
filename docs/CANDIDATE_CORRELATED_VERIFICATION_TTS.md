# Candidate: Correlated Verification for Test-Time Scaling

## Status

**Culled before GPU.**  This is not a paper claim or a positive result.  A July
2026 preprint, *LLMs as a Jury*, occupies the central mechanism and method
space described below.

## Core question

Does a verifier's marginal accuracy/ROC characterize test-time scaling when a
generator produces *dependent* candidates (self-revisions, shared plans,
prompt variants, or other common failure modes)?  If not, can deliberately
diversifying candidates by their failure modes improve verified accuracy at a
fixed generation-and-verification budget?

## Proposed claim

The marginal verifier ROC is sufficient for independent best-of-N candidates,
but not for realistic correlated candidate sets.  For dependent rollouts,
verified accuracy additionally depends on the joint tail dependence between
candidate correctness and verifier errors.  A cluster-then-verify policy that
spends candidates across distinct solution/failure-mode clusters should improve
true selection accuracy relative to best-of-N and naive self-refinement at the
same budget.

This makes a falsifiable, practical statement: more rollouts that share a
verifier-confusing error mode are not equivalent to more independent attempts.

## Nearest work and distinction

| Work | What it establishes | What this candidate must add |
| --- | --- | --- |
| [ROC-n-reroll](https://openreview.net/forum?id=3Gy5mmyuxn) (ICLR 2026) | For independent, fixed-verifier best-of-N and rejection sampling, instance-level performance is characterized by the verifier ROC. | A dependence-aware characterization, an empirical demonstration on correlated/self-refining rollouts, and a policy that improves actual verified selection. |
| [QuasiMoTTo](https://arxiv.org/abs/2607.01179) | Marginal-preserving correlated sampling can improve *oracle* pass@k sample efficiency. | The object here is imperfect-verifier selection, including correlated verifier failure modes; neither oracle pass@k nor generic sample coverage establishes it. |
| Existing best-of-N / self-consistency methods | Increase candidate count or aggregate answers. | Explicitly model and reduce *verifier-error* dependence, rather than token/output diversity alone. |
| [LLMs as a Jury](https://arxiv.org/abs/2607.10139) | Cross-model consensus exploits error decorrelation, gives a parameter-free law with a shared-error floor, and outperforms self-consistency and learned verifiers across math, science, and code. | This directly claims the prospective mechanism, dependence diagnostic, and diversity-based selector.  The candidate is therefore closed. |

The first kill condition triggered: a source already characterizes the relevant
dependence and supplies a more mature equivalent method.  Do not run the gate.

## Initial GPU gate

### Setup

- Generator: Qwen3.5-9B (plus one independent family if the first signal is
  positive).
- Tasks: exact-answer math and executable code, so true correctness is
  programmatically scored; no LLM-as-judge is used as ground truth.
- Candidate-generation policies with equal token budget:
  1. independent samples;
  2. self-revisions from a shared draft;
  3. prompt paraphrases that share a plan;
  4. diversity-targeted proposals.
- Verifier: a separately prompted/frozen open model.  Record its score and
  ground-truth correctness for every candidate.

### Necessary prediction

After matching marginal candidate quality and marginal verifier ROC, the
self-revision/shared-plan conditions must exhibit materially higher
false-positive tail dependence and worse true best-of-N selection than the
independent/diversified conditions.  The proposed cluster-then-verify policy
must recover a meaningful portion of that loss at fixed compute.

### Pre-registered continuation threshold

Continue only if all conditions hold on held-out tasks:

1. The dependent conditions have a pre-specified, nontrivial excess joint
   false-positive statistic (bootstrap CI excluding zero).
2. Their true verified accuracy is at least **5 percentage points** below the
   independent/diversified condition at one or more realistic budgets, despite
   matched marginal ROC.
3. Cluster-then-verify recovers at least **3 points** without reducing oracle
   pass@N or merely shifting compute to a stronger verifier.
4. The sign replicates on a second model family or a second exact-verification
   domain.

If the first condition fails, the thesis is killed rather than reframed as a
generic diversity paper.

## Full-paper surface if the gate passes

1. An exact dependence-aware extension of verifier-based scaling analysis,
   including impossibility of identifying selection performance from a marginal
   ROC alone.
2. A measurable joint-tail diagnostic and a budgeted cluster-then-verify
   algorithm.
3. Cross-domain evaluations (math and code) across two model families, with
   fully programmatic truth labels.
4. Ablations separating generator correlation, verifier correlation, answer
   diversity, and verifier strength.
5. An open reproducibility suite with locked held-out tasks and cost curves.

## Risks

- The theorem may be elegant but too incremental relative to ROC-n-reroll.
- Candidate clustering may be unstable or no better than simple temperature
  diversification.
- A positive effect only in contrived self-revision prompts would not support
  an ICLR submission.

Those are reasons to run the small gate first, not reasons to assume success.
