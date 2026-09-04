# ThoughtTrace collision audit for Hindsight

## PI decision

ThoughtTrace closes the broad claim that next user messages are lossy proxies
for private reactions. Hindsight remains defensible only as a paper about the
different problem that both public messages and private immediate reactions are
post-treatment measurements and therefore cannot identify whether an assistant
satisfied a prior preference or changed the user's persistent state.

Do not market “thoughts are better feedback than messages,” thought-guided
rewrites, or thought-guided on-policy distillation as our contribution.

## Primary source and local provenance

- Paper: [ThoughtTrace](https://arxiv.org/abs/2605.20087), arXiv 2605.20087.
- Public repository: [thoughttrace-project/ThoughtTrace](https://github.com/thoughttrace-project/ThoughtTrace).
- Pinned local source commit: `c3078d310c623f7c28f31e16ed7413730ea0c1e2`.
- Dataset SHA-256:
  `43ed4584661dc78ccd43dceeb84f88ac9780744997e32c5d329bf05292e95d25`.
- Repository license: Apache-2.0; license SHA-256:
  `6343d5757f2b8202379d541a1132ccd047571be188def14f8a98be42fcab5b7c`.
- Local ignored source root:
  `artifacts/thoughttrace_source_20260904_v1`.

Aggregate schema inspection reproduces 1,058 users, 2,155 conversations,
17,058 messages and 10,174 thoughts. It does not print or export participant
text, demographics or identifiers.

## What is already occupied

ThoughtTrace supplies self-reported reasons for user messages and reactions to
assistant responses. The paper reports that reaction-to-next-message pairs have
the weakest semantic coverage and largest embedding separation, and that
frontier models cannot reliably infer the private reaction from dialogue. It
then trains Qwen3.5-4B with thought-guided DPO rewrites and reports stronger
Arena-Hard performance than message-guided rewrites. Its future-work section
explicitly proposes thought-guided reward modeling and on-policy distillation.

The appendix also analyzes thought type versus subsequent conversational
behavior and remaining conversation length. A generic “silent dissatisfaction
causes missing next-turn feedback” analysis on this release would be too close
to those reported results and is not queued.

## The remaining distinction

A ThoughtTrace reaction is elicited after the assistant response. It is richer
than the next message, but it is still downstream of the action. The paper calls
these annotations ground-truth reactions; it does not claim they recover a
pre-interaction preference or distinguish transient expression from persistent
preference transition. Its limitations explicitly note that eliciting thoughts
can itself shape or polarize reasoning.

Therefore ThoughtTrace strengthens the premise that messages omit internal
state while simultaneously showing that immediate private reaction is not the
identifying intervention Hindsight needs. The Hindsight contribution must be:

1. exact observational equivalence of expression and persistent transition;
2. a decision-theoretic consequence for next-turn self-distillation;
3. a declared longitudinal estimand measured by a genuinely delayed, neutral,
   or pre-treatment probe;
4. neural evidence that sparse probes estimate the desired SDPO gradient more
   accurately than equally sparse probe-only learning; and
5. policy-level and human-facing evidence with explicit causal limitations.

## Queue consequence

Run the corrected nested-budget neural gradient G0 first. Run the checksummed
PUPPET capable-reader DEV in the same rental as an independent human-substrate
test. A ThoughtTrace model run is not justified: it would duplicate published
message-versus-thought results without identifying preference transition.
