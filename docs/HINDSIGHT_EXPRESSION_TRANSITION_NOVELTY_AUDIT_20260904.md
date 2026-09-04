# Hindsight expression--transition novelty audit

## PI conclusion

The exact binary construction is useful but is **not sufficient standalone
novelty**.  The general statistical distinction between change in a latent
construct and change in its measurement is mature.  The defensible opening is
narrower: next-turn LLM learning treats an action-dependent user utterance as
privileged supervision even though the same immediate interaction law can arise
from transient expression or persistent preference transition.  In the scoped
search below, no prior work both establishes that ambiguity for next-turn
self-distillation and evaluates an intervention that separates the mechanisms.

Accordingly, retain the theorem as the paper's diagnostic spine, but do not
market it as a new general latent-variable theorem.  A paper requires a faithful
SDPO-style learning result, an anchor/intervention method, and empirical evidence
beyond the constructed binary example.

## Directly adjacent work

- [SDPO from user interactions](https://arxiv.org/abs/2603.12273) conditions on
  the next user message and gives a latent-reward interpretation under idealized
  response and conditioning assumptions.  It demonstrates useful learning from
  interactions but does not distinguish action-induced expression from a
  persistent preference transition.
- [Dynamic Reward MDPs](https://proceedings.mlr.press/v235/carroll24a.html)
  formalize changing and influenceable reward functions and show that static
  objectives can reward undesirable influence.  This occupies the broad claim
  that preferences change; it does not supply an observation/emission channel
  that identifies transition versus reporting.
- [Constructive Alignment](https://arxiv.org/abs/2607.00001) treats human
  preference trajectories as controlled state variables.  It occupies the broad
  control framing, not the immediate-log equivalence or its measurement design.
- [AcCoRD](https://arxiv.org/abs/2608.27818), released August 28, benchmarks
  preferences that are formed, revealed, adjusted, or relaxed during agent
  interaction.  A generic dynamic-preference benchmark is therefore no longer a
  contribution.  Our empirical work must instead test identifiability and the
  consequences of learning from observationally equivalent feedback mechanisms.
- [Causal Preference Optimization](https://arxiv.org/abs/2402.14979) treats
  text-to-outcome optimization as a causal-inference problem and uses randomized
  data plus importance weighting/double robustness.  It addresses confounding
  of exposure and outcome; randomization alone does not separate the two
  post-treatment mechanisms in our construction.

## Older statistical boundary

The psychometric response-shift literature explicitly warns that an observed
post-intervention score can move because the latent construct changed or because
the relation between construct and response changed.  Longitudinal measurement
invariance and repeated indicators are standard tools for making that distinction.
Dynamic discrete-choice work likewise studies identification of structural state
dependence versus persistent unobserved heterogeneity, often requiring additional
variation or multiple observations.  Generic hidden-state transition/emission
non-identification is therefore prior art, not our theorem contribution.

The relevant paper-level novelty, if the experiments work, is the conjunction:

1. connect this measurement problem to next-turn language-model self-distillation;
2. show that ordinary interaction-learning metrics select different policies in
   observationally equivalent expression and transition worlds;
3. identify a declared preference estimand with sparse delayed neutral probes;
4. retain learning from genuine corrections while avoiding expression-induced
   updates; and
5. demonstrate the effect with natural-language users and at least one human
   longitudinal substrate.

## Consequence for the queue

The existing Qwen3.5-9B PUPPET reader DEV remains the cheapest next GPU check.  It
tests whether participant messages contain query-general signal about measured
later belief shift beyond assistant text.  A pass would validate the human-data
measurement substrate; it would not validate SDPO harm or the anchor correction.

If that reader qualifies, freeze and run the untouched 20-query confirmation.
Independently, the next mechanism experiment must use matched expression-only and
persistent-transition user processes with identical first-turn logs.  Compare raw
hindsight learning, no adaptation, anchor-only learning, and an anchor-corrected
hindsight learner under equal anchor and update budgets.  A capable positive
control and zero-influence regime are mandatory.  Do not spend on a broad model
sweep until the correction beats the same-anchor baseline in this small gate.

