# Candidate: outcome-blind process verification

## Status

**Finalist under novelty review. No GPU experiment has been authorized.**

Recent work identifies a sharp, causal failure: reasoning models often accept
invalid derivations when those derivations have a correct final answer, and
patching final-answer representations flips verdicts. The candidate asks a
narrower intervention question that this diagnosis does not answer:

> Can a deployment verifier separate outcome checking from process checking,
> hiding only the explicit final outcome during a process audit, and thereby
> improve detection of invalid derivations without incorrectly rejecting valid
> ones?

The deployed protocol is a bifurcation, not a claim that outcomes are
unimportant: a deterministic outcome checker verifies the final answer, while a
process verifier receives the problem and derivation but the final-answer field
is withheld. The final outcome never becomes a training target or an input to
the process verdict.

## Why it is potentially distinct

The closest diagnostic, *An Enigma of Artificial Reason*, establishes
answer-confirmation bias on VAIR and causally patches answer representations,
but does not evaluate an outcome-blind verifier intervention. Earlier work on
preemptive answer attacks and answer-conditioned trace generation concerns
reasoning generation or distillation, rather than whether an already-produced
trace can be judged more reliably by separating its outcome and process
channels. Partial-answer masking for self-correction is a serious nearby risk;
the related-work audit must show the distinct evaluation target and causal
protocol are not already published.

## Pre-registered G0

Use only unlabelled runner records for model inference; retain process labels in
a local private answer key. For every trace, evaluate the same frozen model and
the same prompt in two paired arms:

1. **Visible:** the explicit final-answer field is shown.
2. **Blind:** only that field is replaced with a sentinel. The problem,
   derivation, decoding settings, item order, and verdict labels are fixed.

The first run uses VAIR invalid traces and VAVR valid controls, with balanced
verdict-label order. It passes only if all of the following hold:

1. Invalid-trace detection improves by at least 10 percentage points.
2. The paired bootstrap 95% lower confidence bound on that gain is above zero.
3. Valid-trace acceptance declines by no more than 2 points.
4. Both arms have at least 98% parseable verdicts.

Then replicate with an independently trained/open model family and an
independent process-verification set. Randomized or reversed labels, a
placeholder-only control, and a version that masks no outcome are necessary
negative controls. A result that appears only under one label interface or only
on an answer-visible benchmark is a failure.

## Kill conditions

- Any nearest work already demonstrates this verifier-side intervention.
- G0 misses any pre-registered criterion.
- An independent verifier, placeholder control, or label permutation removes
  the gain.
- The protocol cannot be paired with a real outcome checker at a competitive
  end-to-end cost.

## Evidence and nearest work

- Sun et al., *An Enigma of Artificial Reason: Investigating the
  Production-Evaluation Gap in Large Reasoning Models*, arXiv:2606.01462.
- Wan et al., *Unveiling Confirmation Bias in Chain-of-Thought Reasoning*,
  Findings of ACL 2025.
- Xu et al., *Preemptive Answer “Attacks” on Chain-of-Thought Reasoning*,
  Findings of ACL 2024.
- Lee et al., *Answer-Conditioned Chains of Thought Degrade Verifiable-Reasoning
  Distillation in Large Language Models*, arXiv:2607.14552.
