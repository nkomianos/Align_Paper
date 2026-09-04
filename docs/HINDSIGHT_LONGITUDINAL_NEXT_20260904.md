# PI scope decision: do not automatically resume report-copying sweep

Independent read-only review after the generation audit: teacher qualification
alone does not justify the old rho.9 copying comparison. That experiment has
fixed preferences and corrupted reports, not persistent preference changes.
It could diagnose learning from contaminated labels without establishing the
distinctive claim in the proposed paper. No longitudinal job is launched here.

## Discriminating intervention

Define initial preference Z0 independently of exposure. Call expected agreement
of deployment actions with Z0 **baseline-preference fidelity**, not universally
correct welfare. Also report current-preference satisfaction and preference
state after exposure stops, measured using delayed neutral probes.

Give each simulated user persistent state. Use three fixed regimes: stationary
truthful reports; expression-only action copying with Z unchanged; persistent
action-induced changes in Z. Match the last two regimes' first-round reporting
laws. Include stable cases; do not choose transition rates after inspecting harm.

Randomize the causal coupling, retaining matched users and exogenous randomness:

- Closed loop: the current learner's actions reach users and cause their next
  feedback/state transition.
- Open loop: a frozen initial policy supplies exposures while a shadow learner
  receives feedback and updates. This blocks updated learner -> future exposure.
- Frozen/no-update: quantify drift under exposure without adaptation. With
  coupled randomness, user states should match the open-loop arm, although
  learner parameters do not update.

Primary contrast is final baseline fidelity in closed minus open loop,
subtracting the corresponding stationary-regime contrast. A negative value
indicates excess fidelity loss under the declared coupling intervention. This
does not identify every form of manipulation. Closed-loop adaptation might
instead protect an initial preference against random exposure: retain that
possibility rather than assume the desired sign.

## Learner and controls

Qualification of a constrained teacher is not qualification of neural
acquisition. Before a neural mechanism run, the same interface and optimizer
must learn truthful choices. Then freeze one loss; do not reopen arbitrary
own-response/full-KL switching. A report-marginal constrained-choice KL is a
possible choice, explicitly not full SDPO.

Direct supervision on semantic reports is indispensable. If only privileged
likelihood training exhibits the effect, narrow the finding to a score/objective
problem rather than claiming a general causal feedback mechanism. A finite-state
or tabular learner can test mechanism orientation cheaply, but is not LM evidence.

Any correction comparison needs anchor-only and ordinary anchor-plus-feedback
learning with the same labels, acquisition cost and tuning budget. Anchors must
have a specified identifying relationship to unobserved states; two unrelated
anchored domains do not identify all domains. Perfect latent-state probes in a
simulator are oracle evaluation, not a deployable estimator.

## Scientific stopping rules

- No clean acquisition: no harm interpretation.
- No excess closed-loop loss or no persistent probe change: bounded null, no
  post-hoc increase of influence until the preferred effect appears.
- No improvement over same-anchor baselines, or correction requires exact
  simulator knowledge: park the correction claim.
- Before paper-facing execution, specify an identifiable correction or a new
  experimentally testable characterization beyond generic preference drift.

An identification/intervention method distinguishing expression from persistent
change is the plausible remaining contribution, not a proven novelty claim.
It needs an unfamiliar transition law and contaminated-anchor sensitivity.
Existing [SLIFT](https://arxiv.org/html/2608.09109v1),
[implicit/causal reward work](HINDSIGHT_NOVELTY_BOUNDARY_20260904.md), and
preference-dynamics formulations are mandatory comparisons. Semantically valid
induced preferences must not be artificially discarded as irrelevant feedback
to manufacture a win over selective learning.

This is a design decision, not an approved GPU queue or a new positive result.
