# PI memo: polite approval is not yet preference shaping

Bounded scientific review, 4 September 2026. No experiment or source changes.
The v3 full-response SDPO positive control remains a separate frozen job.

## Verdict

**Yes: the currently proposed polite-approval channel is primarily a study of
action-dependent, asymmetric feedback corruption.** If it impairs learning,
that alone would not be a new causal contribution, identify persistent preference
change, or substantiate the report's original claim. Its value is limited to
checking whether an already qualified SDPO implementation is sensitive to one
declared feedback channel. We should not spend several more GPU hours treating
that sensitivity as the missing paper.

This does not make the original question unimportant. It means the experiment
needs an observable distinction between **changed reports** and **changed user
state**, plus an intervention that identifies a specified quantity.

## Where prior work already reaches

[SDPO from user interactions](https://arxiv.org/html/2603.12273v1) uses future
user messages as a training signal and supplies both empirical personalization
and a stylized latent-reward interpretation. Demonstrating behavior under a
different feedback law tests that law; it does not refute all reported results.

[Carroll et al., ICML 2024](https://arxiv.org/html/2405.17713v1), sections2–5,
explicitly model AI-influenceable reward dynamics and compare different notions
of alignment under those dynamics. Their discussion also separates an empirical
model from the normative choice of which changing preferences to privilege.
Thus neither “preferences can change” nor “optimizing future satisfaction can
encourage influence” is our new insight.

[Privileged Likelihood Is Not Automatically Value](https://arxiv.org/html/2608.09263v1)
already distinguishes own-rollout feedback dependence, cross-fitted feedback,
the meaning of a likelihood score, and the gradient induced by a training loss.
A generic own-feedback/cross-feedback comparison therefore has a serious
novelty collision.

There is also a directly relevant new empirical predecessor:
[Influencing Humans to Conform to Preference Models for RLHF](https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-stephane_hk_tmlr_2026.html),
listed by its authors as TMLR August2026. It reports three human studies of
interventions on preference expression, explicitly distinguishing expression
from the underlying unobserved reward function. Our expression-only simulator
would not be the first operational treatment of that distinction.

## A minimal mathematical distinction worth making precise

The following is a proposed elementary construction, not a claimed novel theorem.
Let an initial binary preference be Z, an exposed recommendation be A, and
B~Bernoulli(rho) independent. Consider two worlds:

- **Expression world:** the persistent state remains Z. Immediate report O is A
  if B=1 and Z otherwise.
- **Transition world:** the persistent state becomes Z'=A if B=1 and Z otherwise.
  Immediate report is truthful, O=Z'.

Both imply exactly the same law P(O|A,Z). Therefore even randomized A with
positive support, and even measured initial Z, cannot distinguish these worlds
from one immediate follow-up alone. This is stronger than a deterministic-policy
counterexample whose only problem is absent action support, but remains a small
nonidentification construction, not by itself an ICLR contribution.

Persistence is the distinction: under a later neutral measurement that reads
the state without changing it, the expression world reports Z, while the
transition world reports Z'. Without restrictions on measurement or emissions,
repeated ordinary dialogue can still mix the same mechanisms. More passive
messages do not automatically solve identification.

## Minimal distinguishing experiment and estimand

1. Measure a pre-exposure preference M0 using a declared measurement instrument.
2. Randomize a recommendation A and independently randomize exposure R:
   recommendation exposure versus a content-matched neutral reference. Both
   conditions retain the assigned A as an analysis label.
3. Collect immediate feedback O, then delayed neutral/blinded measurement Md,
   at a fixed delay without further agent influence. Include an explicit
   reporting-only intervention as a separate control, where feasible.
4. Compare immediate reported alignment with delayed alignment, and assess
   measurement stability under reference exposure and repeated measurements.

For binary variables encoded as -1/+1, one operational transition-sensitive
estimand is

    tau_d = E[A * Md | do(R=1)] - E[A * Md | do(R=0)].

Under the simple construction with independent balanced A and Z, a perfect
state-reading neutral instrument gives tau_d=0 in the expression world and
tau_d=rho in the transition world. Immediate reported alignment can be rho
in both worlds. The experiment distinguishes persistence, not moral legitimacy.

Identification requires consistency, randomized exposure, no interference,
stable measurement and a justified exclusion that the neutral anchor is not
itself another influencing exposure. With unknown measurement bias, this is
an effect on **the measured anchor**, not identification of a metaphysical
true preference. Pre/post measurements alone do not guarantee those conditions.
Known symmetric measurement noise attenuates the contrast; unknown or
action-dependent contamination requires sensitivity bounds, not an unqualified
causal point estimate. Estimating mediation through latent preference versus
expression generally requires further assumptions beyond randomizing A.

This estimand deliberately avoids calling any preference change harmful. A
separate learning objective may use initial-preference utility, delayed-anchor
utility or another stated choice. They can disagree. Data do not resolve that
normative choice by themselves.

## What a genuinely stronger learning experiment would add

Use matched structured worlds with identical immediate feedback laws but
different delayed state/anchor laws. Freeze those worlds independently of
learner performance. Do not create a simulator where the proposed correction
is privileged by construction and all other learners lack necessary labels.

The substantive test is whether sparse randomized anchor observations permit
a learner or estimator to distinguish the worlds and optimize the declared
anchor-defined objective more efficiently than strong alternatives:

- raw SDPO and an outcome-based learner;
- anchor-only supervised/personalization learning with exactly the same anchor
  observations and information access;
- a standard noise-aware estimator using those same data;
- an oracle-state reference, labeled as extra information rather than a fair
  baseline.

Ordinary cross-fitting of feedback does not provide the same person's unexposed
counterfactual state. It removes one dependence, not latent-state ambiguity.
Likewise, an inverse-propensity weight corrects a sampling distribution only
under its stated estimand; it does not decide which evolving preference is
the right objective. These distinctions must appear before an algorithm claim.

To become publishable, the project needs something beyond this elementary
construction: e.g. useful partial-identification bounds under contaminated
anchors, a sample-efficient intervention design, or a correction with genuine
information-budget-matched gains and external validity. None has been proved
or demonstrated here.

## Allocation decision for this rental window

Complete and analyze the frozen v3 **positive control** if it qualifies. That
can settle whether our small full-sequence training apparatus works. Do not
automatically queue polite-copying arms afterward just to produce an expected
noisy-feedback degradation.

The original top-ranked paper should remain conditional, not falsely declared
dead. But its distinctive causal contribution requires theory and measurement
design work before another GPU campaign. A simulator-only label-noise result,
even with an attractive plot, would not meet that bar. No new implementation,
remote action or expansion was performed for this memo.
