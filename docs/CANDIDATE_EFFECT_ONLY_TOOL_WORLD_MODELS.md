# Candidate: Effect-Only World Models for Tool Agents

**Status:** culled before GPU; retained as a reusable fork-integrity scaffold.

## Cull decision

This line fails the decision standard and must not be run as the proposed
paper.  Centering a generic text/DOM embedding does **not** identify a quotient
of tool-environment futures: additive common-mode structure depends on the
chosen encoder, rather than following from the environment.  The only clean
mechanism test proposed here injected an artificial nuisance, so even a positive
result would not establish a naturally occurring action-selection failure.

Moreover, [Counterfactual Quotient Models](https://arxiv.org/abs/2608.22092)
already owns the central quotient representation and sufficiency theory, while
web-agent world-model work already supplies transition abstraction and simulated
candidate selection.  A tool-agent implementation would therefore be an
application without a defensible new causal object.  The reset/replay collector
and paired-bootstrap evaluator remain useful infrastructure for a future
candidate, but no model training or GPU gate is licensed by this document.

## Claim

For a resettable tool state `s`, an agent often needs to choose among executable
actions `a_1, ..., a_K`, not reconstruct the entire next UI.  Let the encoded
post-action observations be `z_i = f(o'_i)`.  We propose to learn the centered
branch effects

`e_i = z_i - (1 / K) sum_j z_j`,

rather than each absolute `z_i`.  The quotient removes any observation component
shared by the branches: volatile DOM attributes, timestamps, loading artifacts,
and action-independent task state.  A planner ranks candidates from `e_i` plus
the instruction and current state.

The paper-critical prediction is not that a world model can look ahead.  It is
that **same-state branch centering improves counterfactual action ranking and
executed task success at matched model size, data, candidate set, and inference
budget**, especially after an independent source of action-invariant UI variation
is introduced.

## Why this could be distinct

| Closest work | What it establishes | Required distinction here |
| --- | --- | --- |
| [Counterfactual Quotient Models](https://arxiv.org/abs/2608.22092) | Centered action effects are decision-sufficient in controlled physics environments with synchronized counterfactual rollouts. | A practical partial-fork construction and a tool-agent theorem/estimator for language/DOM observations, plus real executable planning rather than state-control experiments. |
| [Web Agents with World Models](https://arxiv.org/abs/2410.13232) | A web agent can predict transition-focused natural-language next-state descriptions and use them for lookahead. | Learn a *relative, branch-conditioned* target and show a causal benefit from common-mode cancellation, rather than a better next-state description. |
| [World-Model-Augmented Web Agents with Action Correction](https://arxiv.org/abs/2602.15384) | World-model simulation and a judge improve web-agent correction. | No generic simulation/judge claim: isolate a representation target that increases discrimination among actions sharing exactly the same pre-state. |
| [LATCH](https://assets-eu.researchsquare.com/files/rs-8912469/v1_covered_bed82fdb-b99a-40d1-abc3-20eb4b4b9d10.pdf?c=1771583547) | Latent lookahead with typed memory for constraint-safe web agents. | The contribution must be counterfactual quotient supervision and a result on action-invariant observation variation, not another latent simulator. |

This remains viable only if the effect target contributes materially beyond a
strong transition-difference abstraction.  Applying CQM unchanged to a web task
would not be enough.

## First feasibility gate

### Frozen setup

1. Collect same-state, multi-action forks from deterministic BrowserGym MiniWoB
   tasks by replaying each prefix from the identical task seed.  Actions must be
   executable and come from the same candidate generator for every condition.
2. Record the pre-state, all `K` actions, and post-states.  Build a paired,
   semantically irrelevant observation perturbation (dynamic attributes /
   timestamp-like text) that is identical across a fork but independently
   resampled across forks.  The unperturbed benchmark remains the primary
   environment; the perturbation is a mechanism diagnostic, not a substitute
   benchmark.
3. Compare equal-parameter models trained on (a) ordinary transition deltas,
   (b) absolute next observations, and (c) centered branch effects.  Each
   obtains the same serialized pre-state, instruction, action candidates, and
   training examples.  The action policy and candidate-generation budget are
   frozen across conditions.
4. Evaluate on held-out seeds and held-out task templates.  Only after a pass may
   the study expand to an independent web environment.

### Pre-registered continuation conditions

All conditions must hold with paired bootstrap 95% confidence intervals:

| Condition | Continuation threshold |
| --- | --- |
| Counterfactual action-ranking accuracy | Effect-only exceeds transition-delta by at least 5 percentage points; CI lower bound > 0. |
| Common-mode mechanism | Effect-only loses at most 2 pp under the paired irrelevant perturbation, while transition-delta loses at least 5 pp more than effect-only; CI for the between-method robustness gap excludes 0. |
| Executed task success | Effect-only improves success by at least 3 pp at the same action/call budget; CI lower bound > 0. |
| Non-degeneracy | Effect-only does not reduce ordinary unperturbed task success by more than 1 pp versus the strongest baseline. |

If any condition fails, stop this line.  Do not reinterpret a representation
reconstruction gain, an artificial-noise-only result, or a single-environment
effect as a paper result.

## If the gate passes

The full study must add:

- a proof of action-ranking sufficiency and a finite-sample robustness statement
  under branch-shared additive nuisance;
- an independent environment with genuinely resettable tool states;
- a stronger language-model world-model family and a separate planner family;
- ablations that distinguish centering from ordinary transition-difference,
  candidate aggregation, more training data, and a larger simulator; and
- released fork logs, task seeds, perturbation implementation, and exact
  evaluation harness.

The immediate next engineering task is a local fork recorder and CPU fixture
that validates branch identity and target construction before any GPU request.
