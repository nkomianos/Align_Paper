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
- [CausalRM](https://arxiv.org/abs/2603.18736) already combines inverse
  propensity weighting, doubly robust imputation, and class-conditional noise
  correction for reward models trained from selectively observed clicks,
  copies, or votes. Generic propensity correction or “observational feedback is
  noisy and selected” is therefore occupied. Its unit is a fixed
  prompt--response pair with a latent ground-truth preference label; the
  observability indicator is the treatment being corrected. It does not model
  an assistant action changing the user's persistent state, nor distinguish
  that transition from a transient next-message expression under exactly
  matched logs. CausalRM must be a baseline/contrast, and our IPW extension is a
  robustness device rather than a novelty claim.
- [Doubly Robust Alignment](https://arxiv.org/abs/2506.01183) already occupies
  generic double-robustness claims for preference/reference-model
  misspecification. Any future augmentation beyond known propensities must be
  stated for the longitudinal post-treatment estimand, not branded merely as
  “doubly robust alignment.”
- [ThoughtTrace](https://arxiv.org/abs/2605.20087) directly establishes that
  private user reactions are richer than next messages, trains thought-guided
  DPO, and proposes thought-guided on-policy distillation. This occupies broad
  “messages are lossy feedback” and “use internal reactions instead” framings.
  Its reactions are themselves measured after the assistant response; it does
  not distinguish prior-preference satisfaction from action-induced persistent
  transition. See the dedicated ThoughtTrace collision audit.
- [Self-Consuming Performative Loops](https://arxiv.org/abs/2601.05184) shows
  bias amplification under iterative LLM retraining and incremental fine-tuning
  with a controlled decision-dependent data mixture. It occupies a broad claim
  that performative feedback can amplify LLM bias, but its human feedback is an
  imposed group-sampling rule; it does not infer whether a next user message is
  expression or persistent state transition.
- [Performative Prediction](https://proceedings.mlr.press/v119/perdomo20a.html),
  [stateful performative prediction](https://proceedings.mlr.press/v151/brown22a.html),
  and [Performative Power](https://arxiv.org/abs/2203.17232) already supply the
  general decision-dependent-distribution, persistent-state, and causal
  influence framings. Neither the broad performativity claim nor a generic
  two-point minimax consequence is standalone novelty here. The remaining
  application-specific question is whether next-turn self-distillation mistakes
  these mechanisms and whether sparse delayed probes correct its neural update.

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

The finite-state method and robustness gates now pass, but their Horvitz--Thompson
machinery is not new in view of CausalRM. The next experiment must test the
application-specific object CausalRM does not: the gradient induced when a
next-turn SDPO teacher receives observationally equivalent expression or
transition messages. The corrected Qwen3.5-9B neural-gradient G0 compares eight
outcome-blind sparse panels at four and eight anchors with the full delayed-
feedback oracle. Do not run a
broad model sweep or claim a new debiasing estimator unless this neural control
variate actually beats equal-anchor estimation.

The capable PUPPET reader remains a separate sub-hour GPU check of whether
participant messages contain query-general signal about measured later belief
shift beyond assistant text. A pass would validate a possible human-data
substrate, not SDPO harm, causal identification, or the correction. Untouched
confirmation stays locked until that reader passes.
