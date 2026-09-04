# Hindsight novelty boundary, refreshed September 4

Decision: do not promote teacher qualification or generic feedback bias into
an ICLR contribution. The original preference-shaping proposal remains open,
but passing an apparatus test would not resolve the novelty requirement.

## Primary-source overlaps

- [SDPO from user interactions](https://arxiv.org/abs/2603.12273) already uses
  next-user feedback for continual personalization. Replicating useful learning
  on stable hidden preferences is a control, not new science.
- [Privileged Likelihood Is Not Automatically Value](https://arxiv.org/html/2608.09263v1),
  sections3.3–3.4 and appendixA.4, separates own-rollout feedback, independent
  donor feedback, partial gradients and utility. It also analyzes vanishing
  corrective gradients for frozen reverse-KL targets. Neither generic feedback
  dependence nor choosing forwardKL instead is a defensible novelty claim here.
- [ImplicitRM](https://arxiv.org/html/2603.23184), sections2.2–3.3 and appendixA,
  models implicit clicks/copies through latent preference and feedback activity.
  Its unbiasedness theorem explicitly assumes correct posterior stratification.
  This is not by itself identification of latent preference transition. But it
  rules out treating all implicit-feedback debiasing as an unexplored space.
- [CausalRM](https://arxiv.org/abs/2603.18736) addresses noisy, selectively observed
  user feedback using a noise model and propensity weighting. Its abstract
  establishes another mandatory distinction from generic causal reward learning;
  detailed algorithmic comparison would require further full-method review.
- [Governing Preference Dynamics](https://arxiv.org/html/2607.00001), section5,
  separates AI actions from interaction structure and sketches constraints on
  evolving latent preferences/beliefs. It is a conceptual/control formulation,
  not empirical proof of our anchor estimator, but broad preference-dynamics
  framing is not novel. Earlier DR-MDP work is already in the journal.

## What our present runs do not establish

The fixed-preference copying simulator changes reports; it does not simulate
persistent preference transitions. First-token formatting, teacher competence,
fixed-target optimization, expression effects and changes in latent user state
are distinct. Only the first three are presently probed by the clean control.
The two-domain sparse anchors have no designed cross-domain identifying
structure. A failure to generalize from them would not refute an intervention
method that actually has such structure.

The earlier sampling-law audit already exhibits the distinction between an
own-response score and a feedback-marginal KL update. Do not rediscover or count
that elementary result as a new experiment. Any extension must exceed the
existing [audit](HINDSIGHT_SAMPLING_LAW_AUDIT_20260904.md).

## Required next contribution, not a promised result

A defensible causal paper must identify a specified preference estimand under
stated, testable-as-far-as-possible measurement/intervention assumptions; show
a benefit over learning from the same anchors directly; preserve genuine
correction learning; and test unfamiliar user dynamics, not only a generator
chosen to favor the correction. Expression-only and persistent-transition
regimes must be separated, with contaminated-anchor sensitivity. No guarantee
of acceptance follows from these ingredients.

Proceed with the inexpensive full-response teacher audit. Do not request a
large GPU sweep until both a reliable clean learner and a meaningful causal
comparison are specified. The notebook-sized toy tests are apparatus work,
not accumulating independent evidence for an acceptance-ready paper.
