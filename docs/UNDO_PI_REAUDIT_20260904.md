# UNDO: semantic and novelty audit before spending more GPU time

This is an offline audit, not a new model run. Earlier evidence and decisions
remain unchanged. The robustness audit observed 87/96 correct for edited
histories, 91/96 for explicit final updates, and 96/96 for canonical, padding,
and counterfactual controls. The 9.375-point difference is real descriptive
evidence, not zero merely because a 10-point engineering cutoff was missed.
It does not yet demonstrate a new useful intervention.

## A foundational correction to the proposed algebra

The report's `add(x); retract(x) = identity` is **not** a universal relation for
our implemented semantics. Our system says that set overwrites and clear
removes the current assignment, without restoring an older value. Starting
with x=prior, setting x=new and then clearing x ends with x absent, not x=prior.
Cancellation is identity only under an initially-unset precondition, or under
a different stack/transaction rollback semantics that we have not implemented.

Correct universal relations for our registers are:

- `set_x(a); set_x(b) = set_x(b)`;
- `set_x(a); clear_x = clear_x`;
- `clear_x; set_x(b) = set_x(b)`;
- `clear_x; clear_x = clear_x`;
- operations on distinct fields commute.

These yield the normal form consisting of the final operation on each field.
Do not discard clear operations when claiming equivalence for arbitrary initial
states. A clear differs from doing nothing when that field was initially set.
`undo_algebra_audit.py` checks the reducer/normal form on 11,718 finite examples
and records the false cancellation counterexample. This is implementation
validation of elementary algebra, not a novel theorem or a neural result.

## What a local-to-global theorem would actually require

A telescoping argument can bound a final distribution disagreement by the sum
of local rewrite errors only when each relation bound holds at every reachable
prefix state encountered along the rewrite path, with a non-expansive suffix
or an explicitly accounted amplification factor. Average error measured on
short training histories does not supply this uniform premise. An arbitrary
model can implement the exact reducer for histories of length <=5 and ignore
clear operations afterward; it passes every short training example yet fails
long cancellation histories. A claimed length-generalization theorem that
assumes uniform long-prefix consistency would assume much of the desired
conclusion. We have not supplied a non-vacuous learning/coverage guarantee.

## Direct primary-source novelty check

[ICF-Bench, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b13d00a62d438856cfe6fbd13b6b2cb8-Abstract-Conference.html)
already evaluates selective in-context forgetting using 2,000 dialogues and
reports interference failures. Another demonstration of obsolete information
affecting responses is not sufficient novelty.

[CCOPD, May 2026](https://arxiv.org/html/2605.30251v1) trains a student against
the same frozen model with canonical context, reports out-of-domain transfer,
and examines self-generated contamination. Calling canonical distillation or
generic history-consistency training new would be incorrect. Its trained
method, not just an unmodified model, belongs in a serious comparison.

## Strongest next experiment, and current recommendation

**Do not prioritize another large synthetic inference-only factorial within
the eight-hour GPU window.** It could explain the original wording sensitivity
but cannot establish a paper contribution. Do not launch arbitrary local-law
training using the false identity above.

If this direction is resumed, test a specific claim: correct local-rewrite
training provides better long-history generalization per training token than
canonical-context distillation, with no extra inference cost. Use the exact
same base model, examples, optimization budget, and seeds for (1) local-rewrite
distribution matching, (2) answer SFT, and (3) canonical-context distillation.
Include unmodified, explicit-update normalization, and executable state
tracking at inference. State tracking is an oracle on formal operations and
is already perfect; a practical advantage must involve raw natural-language
updates where that oracle is unavailable, not pretend to beat its accuracy.

Qualify natural-language update interpretation separately. Hold out relation
compositions and linguistic templates, evaluate 20/60/100 edits, preserve
counterfactual-change sensitivity, and include external ICF-Bench examples.
Count all preprocessing and inference costs. A short-history accuracy ceiling
does not necessarily eliminate distribution-matching gradients, but it makes
ordinary answer SFT an especially weak proposed intervention.

This is not launch-ready: data-matched learning baselines and external task
semantics are not implemented. No honest GPU runtime can be quoted before a
training throughput probe. A speculative 1-2 hour training allocation is not an
estimate. Existing Qwen3-4B pinned revision
`1cfa9a7208912126459214e8b04321603b3df60c` is suitable for continuity, but one
family does not establish generality. The PI recommendation is **park pending
a defensible intervention/theory**, not "all dialogue-retraction ideas fail."
