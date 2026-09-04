# Anchored population model: harm exists, proposed discordant trend does not

Executed 64 designed exact-expectation trajectories: q in {.3,.7}, influence rho
in {0,.3,.6,.9}, feedback-loss weight in {0,.5,.9,.99}, initial policy .49/.51.
Each uses 3,000 logit-SGD steps of size .2. This is a toy model, not an LM result,
parameter-frequency estimate, or novel empirical discovery. Eight related tests
pass. Raw results: `artifacts/hindsight_anchor_phase_20260904_v1/phase.json`.

Independent initial-preference supervision contributes direction q-p; Bayesian
full reverse-KL feedback contributes the previously verified direction. Their
weighted sum is the implemented update. The anchor is perfect and known, not
an estimated causal correction. It supplies benign adaptation at zero influence.

Eighteen grid trajectories lose more than .01 expected initial-preference utility
from their starting policy. That count describes this arbitrary design grid only.
For example q=.3,rho=.6,feedback weight=.9,p0=.51 ends at p=.9078, utility .3369
versus .496 initially. Terminal agreement is .7347, but initial agreement was
.7984: agreement also fell. Finite trajectories are not a proof of convergence
or a phase boundary, especially at feedback weight .99 where anchors learn slowly.

## Crucial structural check

With action-independent copying probability rho and truthful post-exposure
expression, `agreement = rho + (1-rho) * initial_preference_utility` exactly.
Enumerating the four action/preference cells confirms this identity. Therefore
the original hoped-for signature of increasing agreement with declining utility
is IMPOSSIBLE at fixed rho in this model. Comparing high-rho final agreement to
zero-rho initial agreement would falsely manufacture that result.

The harm is objective mismatch, not evidence that the learner has learned to
increase measured agreement. A model with action-dependent influence could break
the identity, but that assumption needs independent motivation and explicit costs;
adding it solely to obtain the desired plot would not establish a contribution.

## PI decision

Keep the exact loss-mapping insight; no GPU expansion and no paper greenlight.
This result sharpens the hypothesis but does not yet exceed known mode-seeking
distillation or dynamic-preference theory. Next work must characterize the actual
feedback semantics and substantiate any action-dependent influence assumption,
not tune this grid until a favorable narrative appears.
