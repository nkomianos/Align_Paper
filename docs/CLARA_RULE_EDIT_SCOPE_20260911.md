# CLARA rule editing: retained property, missing contribution

Reviewed the saved core proposal text and the final proposal's rule-editing,
provenance and tractability passages. The original files remain unchanged and
private. Their locations and historical SHA-256 values are recorded in
`artifacts/osh_clara_audit_20260905/sources.json`. This follow-up concerns generic
perception and logic; it does not implement the proposal's application platform.

## Distinguish three claims before proposing an experiment

| Change | What can remain fixed | What is not established |
|---|---|---|
| A new Boolean or logical composition of available facts | Perception weights and cached facts | Correctness of the facts, distributional reliability, or novelty of rule editing |
| A new predicate inferable by querying the same frozen perception model on retained input | Neural weights | Zero additional inference, accurate new predicates, or sufficiency of old cached facts |
| A new predicate not recoverable from the retained representation | Nothing guarantees a cache-only solution | Neither a rule edit nor a proof certificate recreates discarded information |

This preserves the useful within-vocabulary editability property. It rejects
the stronger inference that every task/rule change can be handled correctly
using the same cached outputs, without new information. The proposal's broad
comparison with pipelines that require retraining for every rule change is not
a fair universal baseline; a modular deterministic rule engine is already a
same-weights comparator.

## Elementary information boundary

Let the retained representation be z=f(x). A rule-specific deterministic answer
h(x) can be computed exactly from z for every input if and only if h is constant
on every fiber of f. Necessity follows because equal z forces equal outputs;
sufficiency follows by defining the answer using any member of each fiber.

For example, if x=(shape,color) but f retains only shape, a new color rule can
assign different answers to two inputs with equal f. No cache-only decoder can
answer both correctly. Retaining the image and rerunning a frozen model is a
different information/computation condition and is not ruled out by this example.
The argument makes no assertion about which facts an actual neural embedding
retains. That must be measured. This is standard information sufficiency, not
a new theorem or an empirical experiment.

## Current primary-source screen

[Flexible Concept Bottleneck Model, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39234)
addresses flexible concept sets and adaptation to unseen concepts. The primary
abstract was inspected, not the full implementation. It is relevant to concept
expansion, but does not by itself establish an exact collision with every
possible rule-edit method.

[Constraint-Based Analysis of Reasoning Shortcuts](https://arxiv.org/html/2604.23377v1)
formalizes alternative concept mappings, verifies uniqueness using ASP, and
proposes constraint repair and label-selection strategies. Its primary abstract,
introduction, related work and method outline were inspected; the full proofs
and empirical results were not independently verified. Generic shortcut
certification and symmetry-breaking label acquisition are already occupied.

[Toward Neurosymbolic Reinforcement Learning via Editable Specifications](https://ojs.aaai.org/index.php/AAAI-SS/article/view/42594)
is an additional primary symposium source on editable specifications. It was
located and its primary landing page opened; no full-paper empirical verification
was performed. Do not turn this limited screen into a universal novelty claim.

## Decision for the existing queue

The exact Bayesian CLARA pilot measures a constructed uncertainty objective
with stipulated facts and probabilities. It does not test neural predicate
acquisition, feature sufficiency under schema expansion, or natural rule-edit
transfer. Its matched-coverage and simultaneous-risk audits remain valid at
their original scope.

No newly qualified CLARA experiment emerges from this source review. A future
proposal must choose within-vocabulary edits or predicate expansion explicitly,
declare whether raw inputs remain available, account for new perception calls,
and compare a frozen perception plus ordinary rule-engine baseline. It also
needs a specific advantage beyond established concept/constraint repair and a
natural evaluation task. These are admission requirements, not an implemented
or scheduled experiment. Do not launch the original unification platform or
another joint-confidence-set demonstration on the basis of this review.

Classification: source/estimand audit, not a neural positive or negative.
Submission remains NO-GO; the original goal remains incomplete.
