# Review of externally proposed low-resource directions

## Decision

**Do not launch a new paper gate from this memo yet.**  Its strongest idea is
important, but it is already pre-empted in its proposed form; the other two are
applications of methods or safety mechanisms that now have direct recent work.
This review does not alter the independently frozen recency-gated G0, which is
the only currently authorized accelerator experiment.

## 1. Activation-space structured dissent for multi-agent sycophancy

The underlying problem is real and high-impact.  However, the proposed novelty
is no longer available.  [Kumarappan and Mujoo (2026)](https://arxiv.org/abs/2605.12991)
already reports the central empirical and mechanistic claims in the memo:

- correct-to-incorrect flips from peer pressure across four model families;
- an attention-dominant L14--L18 causal window;
- activation-space causal analysis; and
- a correctly arguing dissenter that reduces yield across prompt framings.

The paper's proposed mitigation is explicitly *structured dissent*.  Thus a
paper that injects a dissent representation or demonstrates a dissenting-agent
defense would be a replication/implementation of its principal method, not a
new ICLR contribution.  It is additionally crowded by
[The Consensus Trap](https://arxiv.org/abs/2604.17139), which gives a
token-level collaboration defense and a formal robustness analysis for
adversarial majorities.

**PI decision: no green light as written.**  A future candidate would need a
different causal object, such as an empirically falsifiable account of *when a
system can authenticate independent evidence under adaptive source compromise*,
with a defense that is not equivalent to dissent prompting, response voting, or
token-level interleaving.  This is a research lead, not yet a paper hypothesis
or GPU-authorized experiment.

## 2. Information Lattice Learning for molecular prediction

ILL is a genuine, interpretable framework, but “apply ILL to a small molecular
property benchmark and compare it with GNNs” is an application study rather
than a new learning contribution.  The recent
[ILL-as-PGM work](https://arxiv.org/abs/2606.19366) already develops its
structure-learning interpretation, while molecular property prediction has a
large, mature benchmark/method ecosystem.  The memo's energy-efficiency claim
does not by itself establish predictive utility, data efficiency under
scaffold-disjoint splits, calibrated uncertainty, or scientific validity.

**PI decision: do not green-light.**  This could become credible only with a
new theory or a high-value scientific target where an ILL-specific rule is
prospectively validated by external experiments.  We do not have that target or
validation capacity in scope, so a low-resource benchmark comparison would have
low acceptance odds.

## 3. Provenance-aware memory for agent security

The safety problem is important, but the exact design space is already directly
occupied.  [ProvenanceGuard](https://arxiv.org/abs/2607.01236) formalizes
provenance-based intervention before tool execution and evaluates it across
agent benchmarks, while [Towards Verifiably Safe Tool
Use](https://arxiv.org/abs/2601.08012) specifies capability/trust labels and
enforceable data-flow constraints.  A module that tags memory writes by user,
web, or peer source and blocks reads or tool calls is therefore not a distinct
method by itself.

**PI decision: no green light.**  A viable successor would need a new threat
model and a non-obvious guarantee, not a new wrapper around provenance tags.
The prior in-repository provenance-authority gate also failed its predeclared
effect-size threshold, so it must not be revived by changing metrics or labels.

## 4. Data attribution and contamination using STRIDE

[STRIDE](https://arxiv.org/abs/2606.05165) itself proposes activation-space
steering operators for training-data attribution and already evaluates data
contamination as an application.  Reusing STRIDE to show that a benchmark may
be contaminated would therefore be an application/replication unless it solves
a different attribution identifiability problem with a new method and controlled
retraining ground truth.

**PI decision: no green light as an ICLR main-track paper.**  It is useful as
evaluation infrastructure or as a component of a later, genuinely distinct
study, but not as the central contribution.

## Implication for the active program

The memo is still useful: it confirms that mechanistic, causal, low-training
experiments are computationally feasible and points to multi-agent evidence
dependence as a live problem area.  It does not provide a publishable
ready-to-run idea.  The correct sequence is:

1. finish the already preregistered recency gate without altering it;
2. decide from its verified evidence whether that candidate lives or dies; and
3. continue the literature-first search for a new causal object rather than
   spend GPU time reimplementing any memo proposal.
