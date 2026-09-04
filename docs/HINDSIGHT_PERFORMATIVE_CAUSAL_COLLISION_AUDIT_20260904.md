# Hindsight Performative/Causal-Personalization Collision Audit

Date: 4 September 2026. Status: **the lead survives narrowly; generic causal and
sequential-identification claims are occupied.**

## Closest primary work

1. [Aligning Language Models from User Interactions](https://arxiv.org/abs/2603.12273)
   introduces SDPO from user follow-ups. It supplies the exact learning mechanism
   whose causal interpretation is at issue.
2. [Causal Inference out of Control](https://proceedings.mlr.press/v235/cheng24d.html)
   (ICML 2024) already shows that isolated observations need not identify a
   platform's performative effect and gives conditions under which repeated
   sequential observations and responsive control identify it without treatment
   randomization.
3. [Breaking Feedback Loops in Recommender Systems with Causal Inference](https://arxiv.org/abs/2207.01616)
   already formalizes deployment-induced recommendation/response feedback and
   uses intervention distributions and inverse-propensity adjustment.
4. [NextQuill](https://arxiv.org/abs/2506.02368) (ICLR 2026) already markets a
   causal personalization loss: it contrasts predictions with versus without
   user history and emphasizes tokens attributed to user characteristics rather
   than context.
5. [Performative Prediction: Past and Future](https://arxiv.org/abs/2310.16608)
   makes the generic learning-versus-steering distinction and self-confirming
   dynamics established background.

## What is not novel

- Predictions or recommendations can change future data.
- Naive feedback loops can bias a learner or homogenize users.
- One observation may be insufficient for causal identification.
- Sequential measurements or interventions can restore identification under
  assumptions.
- User history and non-preference context should be causally disentangled.
- A generic inverse-propensity, doubly robust or performative-prediction method.

The finite-state expression/transition equivalence theorem should therefore be
presented as the sharp problem instance needed to analyze SDPO, not as a new
general causal-identification theorem.

## Surviving paper claim

The defensible conjunction remains:

1. In next-turn LLM self-distillation, the same immediate natural-language
   follow-up can arise from a transient expression response or a persistent user
   preference transition, while implying opposite persistent-preference updates.
2. The standard SDPO hindsight distribution consumes the post-action message as
   information and therefore cannot resolve those observationally equivalent
   mechanisms from ordinary logs.
3. Sparse neutral delayed probes permit a paired delayed-minus-immediate
   correction that uses the entire ordinary interaction population while paying
   for only a small number of persistent-state measurements.
4. The correction must beat equal-delayed-label SDPO and SFT after actual neural
   optimization, transfer to PAHF-derived natural task surfaces, and retain
   genuine corrections when the immediate response is persistent.
5. Longitudinal human data may establish that immediate feedback and later
   measured state diverge in real conversations, but cannot alone identify
   assistant-caused preference change.

Neither Cheng et al. nor CAFL studies language-conditioned self-distillation or
the expression-versus-transition ambiguity. NextQuill treats user history and
user-written targets as sources of preference attribution; it does not model the
assistant response as treatment that may change the latent preference represented
by the next message. These differences are meaningful only if the neural and
external experiments are positive.

## PI decision

Keep Hindsight at **conditional yellow**, not green. Do not submit a theorem-only
or benchmark-only version. A qualified Qwen gradient G0, policy G1, EndoPAHF G2
and independent family or human validation are necessary to overcome the dense
performative-prediction and personalization prior art. If the neural correction
does not beat equal-label baselines, retire this paper rather than falling back
to generic causal language.

