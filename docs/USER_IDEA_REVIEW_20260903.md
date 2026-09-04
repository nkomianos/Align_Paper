# Five proposed papers: PI review and bounded implementation

Reviewed 3 September 2026. **Conditional investigation of ideas 1 and 2, not a
paper green light.** Neither a simulator programmed to exhibit influence nor a
successful prompt correction establishes a new learning algorithm. No GPU was
contacted, no language-model weights were updated, and no human data were
acquired for this review.

The official [ICLR 2027 call](https://www.iclr.cc/Conferences/2027/CallForPapers)
confirms abstracts on **18 September, 23:59 AoE**, and full papers on **25
September, 23:59 AoE**. The [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
require the author list at the abstract deadline. The dates are real; they do
not make an unvalidated hypothesis acceptance-ready. No acceptance probability
or full-study GPU-hour estimate is justified yet.

## Portfolio decision

| Proposal | Judgment | Immediate commitment |
| --- | --- | --- |
| Hindsight Is Not Counterfactual | Best conceptual lead, **conditional**. The identification question survives; the proposed polarization theorem does not follow from the stated model. | Executable finite-model audit, anchor-estimator checks, frozen language-signal assay. Actual SDPO training remains a separate step. |
| Dialogue Retractions as Algebra | Useful bounded backup, **conditional**. Semantics must be explicit; canonical-context distillation is a close baseline. | Executable register semantics, 4–100-update matched dialogues, read-only residual-effect assay. No local-relation training yet. |
| Learned-transport certificates | Park. Must specify what is certified and prove a target-linked non-vacuous inequality. | Analytical counterexample; no paid experiment queued. |
| Succinct but Unlearnable | Park. A question, not yet a precise theorem or experiment. | Require task family, architecture, learner and lower-bound assumptions first. |
| Causal steering subspaces | Do not queue in its present form. Especially high collision risk. | Existing work already discusses identification assumptions, not just impossibility. Need a substantially different theorem or operational result. |

The new cases are added to [the executable queue](../configs/research_queue_20260903.json),
alongside the existing motion, visual-hindsight and latent-channel pilots.
They do not replace or restart any historical experiment.

## 1. Hindsight: what remains genuinely interesting

[Aligning Language Models from User Interactions](https://arxiv.org/abs/2603.12273)
and its [public implementation](https://github.com/lasgroup/user_interactions)
provide a concrete interaction-learning target. Public implementation revision
inspected: `3b17d2a67bd2565b9fbda495fd16a485406aa954`.
Crucially, the paper's idealized analysis **already allows the next message to
depend on the assistant action**. We must not describe it as assuming an
action-independent user. The unresolved question we would test is whether that
signal identifies a *separately specified* prior or delayed preference objective.
Our assay uses a simplified hindsight prompt, not an upstream SDPO reproduction.

[Dynamic Reward MDPs](https://proceedings.mlr.press/v235/carroll24a.html)
already formalizes preference influence and competing objectives. Thus
"feedback is endogenous" and "preferences can change" are not sufficient
novelty. A credible contribution needs a useful identification result and a
correction that beats annotation-budget-matched alternatives in actual learning.

### A compact, exact observational-equivalence construction

Consider independent one-step episodes at a fixed observed context. Let the
initial binary preference be U, response A, and next expressed choice O.
Within an episode the logging policy does not observe U. Two worlds are:

* **Expression world:** P(U=1)=0.7; U never changes. Expression copies A with
  probabilities c_0=0.75 and c_1=0.2, otherwise reporting U.
* **Transition world:** P(U=1)=0.2; expression is truthful. The state copies A
  with probabilities rho_0=0.125 and rho_1=0.7, otherwise retaining U.

Both yield exactly

\[
P(O=1\mid A=0)=7/40,\qquad P(O=1\mid A=1)=19/25.
\]

Proof: in either world the channel is `(1-copy_probability)*q +
copy_probability*a`; substituting the four rational numbers gives those same
two entries. Multiplying by any common randomized logging distribution gives
the same joint law of (A,O). Product laws remain identical for independent
episodes. This is **not** merely absence of action coverage under deterministic
logging. Yet initial-preference utilities for actions (0,1) are (0.3,0.7) in
the first world and (0.8,0.2) in the second: policy rankings reverse.

Limitations: this does not prove equivalence for arbitrary repeated-user
longitudinal designs, and pre-treatment measurements would distinguish these
worlds. The construction is elementary and is not asserted to be a novel
impossibility theorem by itself. It establishes the kind of gap a stronger
paper must address.

### The proposed polarization claim is not automatic

Scope clarification from the subsequent
[sampling-law audit](HINDSIGHT_SAMPLING_LAW_AUDIT_20260904.md): the calculation
below is for the sampled own-response stopped-advantage update. Under endogenous
feedback it must not be substituted for the full reverse-KL update. The new
exact example finds opposite directions; neither is automatically welfare.

Let K_a(o) be a fixed action-to-message channel, p=P(A=1), and
M_p(o)=(1-p)K_0(o)+pK_1(o). Under an exact Bayesian hindsight teacher,

\[
\log\pi(a\mid o)-\log\pi(a)=\log K_a(o)-\log M_p(o).
\]

The expected log ratio equals mutual information I_p(A;O). The expected
stopped-teacher score-function update in logit(p) is

\[
p(1-p)\{\mathrm{KL}(K_1\Vert M_p)-\mathrm{KL}(K_0\Vert M_p)\}.
\]

For the symmetric channel P(O=1|A=0)=0.1 and P(O=1|A=1)=0.9, the gradient is
**+0.109177 at p=0.1**, zero at 0.5, and **-0.109177 at p=0.9**. Balance is
attracting, not polarizing. A finite-difference test also verifies the
mutual-information derivative. This result is conditional on the fixed channel
and ideal teacher; it does not establish safety of deployed SDPO or all updates.

Likewise, a simple population imitation recurrence

\[
m'=(1-\rho)m+\rho b,\qquad b'=(1-\eta)b+\eta m'
\]

has eigenvalues 1 and (1-rho)(1-eta). At rho=0.4, eta=0.3, a policy imbalance
of 0.01 with m=0 converges to approximately 0.006897, not to +/-1. For balanced
initial users and an unpersonalized policy, expected initial match is always
0.5. A falling-welfare/polarization figure therefore requires additional
structure, not just these equations. We will not manufacture it with a
hand-chosen nonlinear update and label it an SDPO finding.

### What anchors identify, under which assumptions

In the binary copying model, suppose q=P(U=1) is measured before treatment,
d_a is the mean of a truthful neutral delayed measurement, and r_a is the
immediate expression mean under randomized A=a. Then

\[
\rho_0=(q-d_0)/q,\quad \rho_1=(d_1-q)/(1-q),\quad
c_a=(r_a-d_a)/(a-d_a).
\]

This requires q in (0,1), nonzero expression denominators, the specified
transition/emission model, and an anchor that measures the intended state
without its own unmodeled treatment effect. Randomizing A alone is insufficient.

The implementation also checks the standard cross-fitted AIPW score for a
delayed binary-anchor ATE. With covariates X, action propensity e(X), anchor
acquisition indicator S, acquisition probability s(X), and outcome models mu_a:

\[
\mu_1-\mu_0+\frac{S}{s(X)}\left[
\frac{A}{e(X)}(Y-\mu_1)-\frac{1-A}{1-e(X)}(Y-\mu_0)\right].
\]

Assumptions are action exchangeability, overlap, anchor missingness at random
conditional on X, independent/cross-fitted nuisance estimates and a well-defined
anchor outcome. Double robustness requires correct outcome models **or correct
joint action/measurement propensities**. It does not protect against arbitrary
anchor corruption or make initial preference uniquely morally privileged.
This is a standard estimator, not a claimed new algorithm.

CPU audit: 12,000 simulated people per regime, 15% sampled anchors, seed 90326:

| Regime | True delayed-anchor ATE | Cross-fitted estimate | Immediate-expression ATE |
| --- | ---: | ---: | ---: |
| Static | 0 | -0.0449 | 0 |
| Expression only | 0 | 0.0044 | 0.70 |
| Transition only | 0.50 | 0.4925 | 0.50 |
| Mixed | 0.50 | 0.5065 | 0.85 |

These are estimator calibration checks with sampling noise, **not LM results**.
Static-arm error is about 4.5 percentage points in this realization. The audit
also tests deliberately wrong nuisance/propensity models. If each arm's anchor
mean can be biased by at most delta, an elementary conservative ATE interval is
`[estimated_ATE-2*delta, estimated_ATE+2*delta]` clipped to [-1,1]. Even a measured
0.2 effect loses a guaranteed sign at delta=0.1. No novel sharp bound is claimed.

### Implemented language-signal assay and remaining work

Prepared data: 128 matched blocks, four input arms each, **512 full forwards**.
Four surfaces (software, travel, schedule, product), two initial preferences,
two logged actions, four dynamics labels, two language styles. The **64-forward
smoke** is a frozen subset. Arms are base, next-turn feedback, independently
collected pre-interaction anchor, and feedback plus anchor.

The same text is used for expression-only and transition events when their
immediate report is identical. Private labels differ; the model cannot see the
latent state. These are controlled realized events, not sampled on-policy
trajectories. The evaluator measures the actual next-token log-ratio on the
logged action, correction comprehension, and how independent anchor information
changes the conflicting-feedback signal. It audits unrestricted choice-token
mass rather than silently renormalizing a noncompliant answer into success.

Frozen full-assay prerequisites: mean A/B probability mass >=0.5; anchor-only
initial-choice probability >=0.8; truthful correction initial-choice probability
>=0.8. A signal requires a wrong logged-action raw log-ratio >0.1 nats and
>=0.1 restoration of initial-choice probability when the anchor is supplied.
Smoke is always `SMOKE_ONLY`; it never earns paper expansion. These thresholds
are triage heuristics, not significance tests or estimates of real-world risk.

**Not implemented:** on-policy SDPO optimization, a learned influence-constrained
policy, Anchor-SDPO, a multi-turn preference learner, human validation, or the
claimed 5–10-point *training-induced* welfare decline. Prompt conditioning on an
anchor is not the proposed training correction. A signal would justify an
upstream SDPO reproduction and a frozen small-model learning experiment, not
the full 90–150-hour matrix.

Before such training: define the anchor estimand; reproduce upstream on a
static-user control; compare no adaptation, genuine SDPO, immediate-feedback
SFT/DPO, anchor-only SFT with matched annotation budget, random-anchor control,
and an offline latent-state oracle. Test zero influence, expression-only,
transition and unseen dynamics; vary anchor contamination. Require recovery of
genuine corrections and independent seeds. A full-model welfare claim requires
on-policy evaluation, not averages of these forced conflict examples.

[Talk2AI](https://arxiv.org/abs/2604.04354) is a relevant external-validation
lead: its paper describes 3,080 conversations from 770 adults over four sessions
and longitudinal measurements. Only publication metadata was checked; data
access, license and usable fields remain unverified. No raw human conversations
are local. Even with access, associations in these logs would not identify the
causal effect of individual assistant messages.

## 2. Dialogue retractions: semantics before a theorem

[Canonical-Context On-Policy Distillation](https://arxiv.org/abs/2605.30251)
already targets disagreement across conversational paths using canonical-context
teachers. Generic history invariance, canonical summarization or teacher/student
distillation is therefore not enough. Belief revision and dialogue repair also
have longstanding literatures. The plausible opening is a demonstrably useful
*local-relation training procedure that generalizes to much longer edit paths*,
not a new name for an old consistency metric.

### Define the state machine correctly

The implemented system uses last-write-wins registers. `set(k,v)` overwrites;
`clear(k)` removes the current value, giving UNSET. It does not restore a stack.

* `set(k,v); clear(k)` is an identity **only if k was absent beforehand**.
* `set(k,v); set(k,w)` is equivalent to `set(k,w)`.
* Operations on distinct registers commute.

Calling retraction a stack-based undo would invalidate the overwrite law when
future undo operations reveal earlier values. The current corpus excludes
branch/merge and models no such stack. Those are not promised as implemented.

### Conditional local-to-global bound, not a theorem breakthrough

Suppose each valid generating rewrite changes the latent represented state by
at most epsilon, uniformly over all reachable prefixes; every suffix update is
L-Lipschitz in a state metric; and the output decoder is K-Lipschitz to total
variation. A derivation of m local rewrites with suffix lengths ell_j satisfies

\[
\mathrm{TV}(P_h,P_{h'})\leq
\min\{1,K\epsilon\sum_{j=1}^{m}L^{\ell_j}\}.
\]

Proof: one local discrepancy is at most epsilon, propagated through its suffix
by repeated Lipschitz bounds, then through the decoder by K. Sum the resulting
adjacent-history discrepancies by the triangle inequality. This standard
telescoping argument is not asserted to be novel. It can be vacuous when L>1 or
the derivation is long; finite short-history training does not establish the
uniform reachable-prefix premise. A publishable theorem must do more, or the
training result must be unusually compelling against strong existing baselines.

### Implemented assay

Four surfaces (software requirements, travel, scheduling, fictional permission
registers), three valid relations, depths **4,20,60,100**, four repeats, four
arms: **768 forwards** full, **48 smoke**. Arms:

1. Short canonical terminal state.
2. Expanded update dialogue with fixed assistant acknowledgements.
3. Same terminal state plus matched-depth irrelevant operations.
4. A one-operation counterfactual that changes the required answer.

The third arm controls edit count, **not exact token count**. All actual
token lengths are audited before inference, with no truncation allowed. The
fourth tests sensitivity to the consequential update. Fixed acknowledgements
mean this is not yet a fully interactive agent benchmark. Permission registers
are fictional text only; no real authorization operation is executed.

At every depth the full-assay validity requirements are canonical accuracy
>=0.9, matched-depth accuracy >=0.8, and counterfactual accuracy >=0.8, plus
overall choice-token mass >=0.5. A residue signal requires, at some depth >=20,
a >=0.1 matched-depth-minus-expanded accuracy gap and >=0.05 excess probability
on the stale answer. Stale-answer metrics exclude the commutation case, whose
alternative value was never asserted. These are exploratory triage criteria,
not adjusted statistical evidence for all lengths/surfaces.

**Not implemented:** algebraic local-relation training, CCOPD reproduction or
length-generalization training comparisons. If the apparatus and signal survive,
compare short-relation training against token/compute-matched ordinary SFT,
canonical summarization, CCOPD-style distillation, and an executable state
tracker. Train on short identities only and reserve genuinely unseen long
histories/templates. If a state tracker solves everything with comparable
cost, a model-training contribution needs a compelling use case beyond it.

## 3–5. Why they do not receive GPU slots

**Transport:** in two dimensions, let the source and target both be standard
isotropic Gaussian. The identity and a 180-degree rotation are smooth-flow
maps with identical output distributions and exact inverses. Both have zero
round-trip defect and zero endpoint distribution discrepancy, but their paired
endpoint mean-square difference is 8. Thus these measurements cannot certify
agreement with a designated coupling. This does not make the rotation a bad
sampler; it makes the proposed correctness target ambiguous. Define a
test-function/distribution or numerical-integration guarantee first, then prove
the target-linked inequality. Do not sell cycle consistency as correctness.

**Succinctness:** [Transformers Are Inherently Succinct](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5f7804e8855efe5554025217abc49315-Abstract-Conference.html)
motivates a question about learning, not the requested separation itself.
Without a specified optimizer, initialization, precision, task distribution and
complexity assumption, a toy training failure would not prove unlearnability.
No speculative GPU sweep is queued.

**Steering:** [On the Identifiability of Steering Vectors in Large Language
Models](https://arxiv.org/abs/2602.06801) discusses independence, sparsity and
multi-environment/cross-layer conditions. Those substantially overlap the
suggested extension. Do not claim those ingredients as an unoccupied direction.
This is a scoped collision judgment, not an exhaustive novelty guarantee.

## Reproduction, queue and readiness

New source: `src/interaction_sprint/{theory,fixtures,run,analyze}.py`.
CPU audit archived at `artifacts/interaction_sprint_theory_20260903_v2`;
earlier v1 preserved. Prepared datasets and their private keys are local,
ignored artifacts, reproducible deterministically from source:

* `artifacts/endo_signal_20260903_v1`: 64 smoke / 512 full.
* `artifacts/undo_algebra_20260903_v1`: 48 smoke / 768 full.

Both use pinned `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c` as a low-cost reference apparatus,
not as evidence about the newest model families. First smoke total is **112
forwards**, zero training steps. Local tokenization with the pinned real
tokenizer verifies single-token A/B/C/D choices (IDs 32/33/34/35), all 512
hindsight inputs at 37–75 tokens, and all 768 UNDO inputs at 105–3,022 tokens.
No weights were downloaded for that check. Actual GPU minutes must be estimated from those
runs; the user's full-paper compute estimate is not a measured gate cost.
Different-length UNDO prompts make an extrapolation from only short smoke
prompts unreliable; benchmark one long-context example before budgeting full.

Preparation and CPU audit (always choose new output paths):

```powershell
$env:PYTHONPATH='src'
python -m interaction_sprint.theory --output artifacts/interaction_sprint_theory_NEW
python -m interaction_sprint.run prepare --study endo_signal --output artifacts/endo_signal_NEW
python -m interaction_sprint.run prepare --study undo --output artifacts/undo_algebra_NEW
python scripts/check_research_queue.py
python -m pytest tests/test_interaction_sprint.py -q -o addopts=''
```

Remote launch contract: transfer and verify the prepared root, use an audited
clean source commit and an isolated CUDA environment, then set
`SPRINT_PYTHON`, `SPRINT_PREPARED`, `SPRINT_RUN_ROOT` (fresh absolute path),
`SPRINT_PINNED_COMMIT`, and `SPRINT_MODE=smoke` before calling
`scripts/run_interaction_sprint_remote.sh`. Never substitute an old frozen
environment's dependencies. CUDA/model loading remains untested. No automatic
expansion, scheduling, or external launch is configured by this change.

The runner saves source, inputs, model/runtime metadata, complete token budgets,
raw log probabilities and choices, completion/failure markers and SHA-256
inventory. Retrieve all evidence, compare remote/local manifests, then:

```powershell
python -m interaction_sprint.analyze --root retrieved/NEW_RUN --output analysis/NEW_REPORT.json
```

The output must be outside the immutable evidence root and must not already
exist. A successful checksum verifies bytes, not causal validity. A successful
assay tests prerequisites, not ICLR acceptance. Keep the initial sprint scoped
to deciding whether the proposed *learning* experiments are worth implementing.
