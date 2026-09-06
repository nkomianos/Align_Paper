# Research shortlist under a 50 H200-hour budget

5 September 2026. Research preparation, not paper qualification. No new model
experiment or remote host access occurred for this review. Primary papers and
author code were inspected; their claimed experimental results were not independently
reproduced. A limitations paragraph is a lead, not proof that nobody solved it.

## What the latest literature changes

| Primary source inspected | Relevant result or explicit limitation | Consequence for us |
|---|---|---|
| [Denser ≠ Better: Limits of On-Policy Self-Distillation for Continual Post-Training](https://arxiv.org/html/2607.01763), July 2026, full HTML sections | Studies forgetting, distributional drift and collapse as artifact amplification in continual self-distillation. | Generic teacher-feedback collapse or formatting amplification is already occupied. Our three-arm run is calibration. |
| [Rethinking On-Policy Self-Distillation for Thinking Models](https://arxiv.org/html/2607.05184), July 2026, discussion/limitations | Links degraded long-budget reasoning to changed behavior at branching tokens. Authors explicitly limit the causal isolation and do not provide a solution. | A controlled causal intervention could be consequential, but merely masking entropy or preserving reasoning markers is not enough. |
| [One Symptom, Three Levers](https://arxiv.org/html/2608.25936), August 2026, review | Organizes collapse around signal placement, privileged context and teacher dynamics. | Do not advertise this decomposition itself as our contribution. |
| [Learning from the Future](https://arxiv.org/html/2607.27055), July 2026, method | Uses future-aware sequential recommendation teachers, a reachability gate and momentum averaging. | Frozen/EMA teachers and privileged-information gates require strong direct comparison; neither is a new method by itself. |
| [SR-OPSD](https://arxiv.org/abs/2608.09745), August 2026, abstract | Proposes self-reference regularization in on-policy self-distillation. | A generic reference-anchoring fix is occupied. |
| [User Feedback Provides a Unique Signal that LLMs Cannot Detect](https://arxiv.org/html/2609.02859), September 2026, methods and appendices | Compares feedback-assisted revisions with revisions without feedback and finds discrepancies between issue resolution and pairwise judging. Its issue-resolution evaluator receives feedback, whereas the default pairwise judge has a different information set and task. | Interesting evaluation target, but another "judges miss feedback fixes" paper would duplicate the headline. Need objective independent ground truth and controlled evaluator information. |
| [Author feedback-blindspot code](https://github.com/shachardon/feedback-blindspot), README | Already exposes feedback-aware evaluation, category-aware evaluation and local model options. | Simply adding feedback to the judge is also not our novelty. Release availability is verified; license and individual dataset fields still need audit before adopting it. |
| [Reliable Post-Retrieval Assembly for Agent Memory](https://arxiv.org/html/2606.01435), current full HTML | Separates extraction from deterministic policy execution. Limitations name partial orders, causal dependencies, historical questions and aggregation; natural LongMemEval evidence is small and not an overall win. | Narrow ambiguity-aware action selection is a candidate; generic deterministic freshness resolution is not. |
| [TOKI](https://arxiv.org/abs/2606.06240), June 2026, abstract | Formalizes contradiction resolution using bitemporal operators and concurrency/isolation contracts. | Bitemporal memory and concurrency contracts alone are occupied. |
| [Fresh Memory, Stale Plans](https://arxiv.org/abs/2609.03340), September 2026, abstract | Checks action dependencies against updated records and blocks/replans obsolete plans. | Revalidating plans or tracking dependency versions is not an open invention. |
| [Dependency-Guided Rollback Repair](https://arxiv.org/abs/2608.10502), August 2026, abstract | Repairs downstream memory/action consequences while preserving unaffected work. | Reject generic provenance-aware selective rollback. |
| [SleepGate](https://arxiv.org/abs/2603.14517) and [Human-Inspired Memory Architecture](https://arxiv.org/abs/2605.08538), abstracts | Already propose sleep-style consolidation, forgetting and/or reconsolidation mechanisms. | Biological names do not create novelty. Demand a measurable prediction distinct from replay, decay and ordinary retrieval. |

The closest existing Hindsight comparisons remain SLIFT, PUMA, Privileged
Likelihood, ordinary supervised learning and prediction-powered residual estimators.
The earlier theory/literature audit supplies these comparisons; the new sources
above make the generic collapse/feedback/memory fallback even narrower.

## Ranked work, including what is NOT promising yet

### 0. Required calibration: learning from reliable labels

Run the prepared three-arm diagnostic once, with a one-hour hard cap. It establishes
whether the learning system works at all. A successful calibration does not outrank
the novelty requirement. A failed current teacher alone is not an ICLR thesis.

### 1. Conditional continuation: useful proxy feedback under sparse trusted supervision

Question: can we determine when abundant interaction feedback improves actual
held-out decision quality over the same small trusted-label set alone, and shut
it off when it is harmful? This retains our strongest finite-model signal.

Current novelty grade: weak. Cross-fitting, residual control variates, learned
mixing and active labeling have established statistical precedents. There must
be a specific interaction-learning failure and a useful advance beyond applying
those methods under another name. Do not claim recovering latent human welfare.

Necessary next screen AFTER calibration/task review: three trusted-label budgets,
matched-label SFT, pooled supervision, a tuned mixture, a suitable established
residual estimator and any proposed method. Freeze the validation budget used
to choose weights, counting it as trusted supervision for every method. Use the
same base tasks and three fixed seeds only after a first-seed improvement.
Stop if the advantage vanishes against tuned mixtures or requires extra labels,
test-dependent hyperparameters, or an unvalidated target. Initial cap: 2 H200 h;
any result is developmental pending independent task/family confirmation.

### 2. Best alternative to investigate on CPU: action-relevant ambiguity in memory

Proposed question, not a discovered result: when several memory updates are
incomparable, can an agent answer without clarification exactly when the possible
resolutions agree on the requested action, and ask a minimal useful clarification
otherwise? The proposed distinction is decision-relevant unresolved evidence,
not storing versions, detecting staleness or blindly choosing the newest entry.

Multidisciplinary inspiration: distributed-systems partial orders and certain
answers in databases, combined with active sensing's cost of observation. These
are existing concepts, NOT novel mathematical inventions. The possible AI
contribution is an effective learned extraction/decision interface and a measured
accuracy-versus-clarification improvement under realistic incomplete metadata.

CPU feasibility first: enumerate small partial orders, demonstrate that arbitrary
linearization can change an action, and identify cases where it cannot. Compare
against ordinary max-timestamp, abstain-on-every-conflict, deterministic certain-
answer reasoning, and a question-conditioned deterministic resolver. No neural
superiority is possible merely by giving our method the gold graph.

First proposed model screen: 240 independently generated base scenarios with
unique resolution, action-irrelevant ambiguity, action-changing ambiguity and
historical questions equally represented. Same retrieved evidence and extracted
candidate set for all resolvers; also a separate end-to-end noisy-extraction view.
Counterbalance wording/order within base, never count these as independent N.
Freeze output parser and charge clarification and extraction calls. Continue only
if a method beats the strongest deterministic resolver by at least 5 percentage
points at matched clarification cost, with no harm in unique-resolution cases.
Cap: 1 H200 h. This threshold is a routing choice, not a power claim.

Kill before GPU if the necessary novelty already appears in TOKI, PlanFence,
certain-answer/query-repair work, or if no external natural workload can be
assembled without fabricating its ambiguity. The synthetic generator alone is
not a sufficient ICLR benchmark. Current status: hypothesis/design only; no new
dataset, resolver, model result or scientific positive is claimed.

### 3. Secondary alternative: objective selection regret under feedback-aware judging

Proposed question: does feedback-aware selection choose fixes that actually
improve externally checked behavior, or just fixes that agree with the feedback?
Use correct, incorrect and irrelevant feedback; hold response pair, style,
candidate order and evaluator information fixed across comparisons. Separate
whether the feedback introduces a new requirement from whether it diagnoses a
violation of an existing requirement. Otherwise "ground truth" changes between
conditions and the experiment is uninterpretable.

Use executable constraints with separate regression tests as independent ground
truth. No LLM-generated issue-resolution verdict may also define correctness for
the comparison being evaluated. Select tasks before seeing which method wins,
and report selection regret on ALL tasks, not only a favorable successful-fix
subset. Baselines: ordinary pairwise judge, feedback-aware pairwise judge,
criterion decomposition, executable checks alone and a stronger same-information
judge. The proposed method still needs to be specified; this is not a prepared
training queue.

First proposed screen: 200 fixed response-pair/task bases with crossed feedback
conditions, two local judge families, saved raw outputs. Cap: 1 H200 h. Continue
only if there is a reproducible failure beyond information asymmetry and a remedy
that beats checks-alone and ordinary feedback-aware judging at matched cost.
Kill if it reduces to already-known judge bias, instruction following or giving
one judge additional criteria. Transfer beyond executable tasks is an unresolved
burden, not an assumed outcome. Code/data availability is promising; actual
license, source units, and independent truth must be checked before adoption.

## Biological inspirations screened out or translated

Homeostasis suggests bounding drift; EMA teachers and reference regularization
already implement close mechanisms. Immune-system memory suggests preserving
trusted exemplars; that reduces to replay unless a distinct prediction survives
matched memory and compute. Reconsolidation suggests making a retrieved trace
editable only under validated contradiction; prior memory systems already cover
parts of this. Active sensing suggests querying only action-relevant uncertainty;
that motivates candidate2 but must beat classical value-of-information methods.
No biology analogy is evidence of efficacy or sufficient novelty.

## Decision and paper gates

Prepare calibration now, without GPU contact. Spend CPU effort on candidate2's
remaining novelty/data feasibility, with candidate3 as a backup. Do not run all
three research lines in parallel on rented compute. A maximum of4 H200 h is
reserved for initial candidate screening; retire failures and choose one.

Before main experiments require: one concrete falsifiable claim, a meaningful
external task, equal-information baselines, capable interfaces and a measured
effect worth replicating. Before manuscript production require a qualified main
result plus at least an independent replication plan with accessible data and
budget. By September8 choose a contribution/data route or explicitly move beyond
this ICLR cycle; by September11–12 require replicated evidence, not a formatted
draft. No candidate currently supports a high-confidence acceptance prediction.
