# Paper pitch after the independent audit

5 September 2026. Research proposal, not a results abstract. Companion:
[skeptical rejection memo](HINDSIGHT_REJECTION_MEMO_AFTER_AUDIT_20260905.md).

**Working title: When Immediate Agreement Cannot Identify Persistent Preference.**

An assistant can receive the same immediate agreement when its response changes
what a user expresses and when it changes a preference that persists. A learner
trained on those logs therefore needs an explicit target and additional
measurement assumptions before claiming to improve persistent preferences. The
proposed paper would make this ambiguity precise and test whether sparse delayed
measurements improve a useful learner beyond ordinary methods given the same
measurements. It would not equate agreement with welfare or define the desired
preference by whichever measurement makes a correction work.

The existing mathematical contribution is an exact two-world witness. Immediate
observations have identical laws, while delayed outcomes separate the worlds and
reverse the ranking of population actions. The general expected-risk minimax
bound is `2/15`. Its regret application concerns a new anonymous user whose
baseline state is unavailable at deployment; observing that individual's state
eliminates regret in this witness. A known full-rank emission model gives ordinary
linear identification. These results motivate a measurement question; they are
not a general impossibility theorem for personalization or evidence that
adaptation automatically polarizes users.

The empirical contribution still has to be earned. Existing finite-state screens
show that a standard residual correction can help in some cells. They also show
that an ordinary anchor mean can beat the IPW version. There is currently no
qualified neural method result or validated external persistence instrument.
The next DEV comparison therefore gives every sparse learner the same 64 delayed
labels and pits correction against pooled SFT, pooled SDPO and a fixed
nonnegative mixture. It measures delayed-label predictive utility directly,
with acquisition, answer-format and paired position controls. This is an
offline prediction test, explicitly distinct from action-induced state utility
in the theorem. A separate conditional semantic gate excludes formatting-only
gains; a matching cheap memory/profile readout stops neural superiority claims.
A qualified negative ends this estimator path.

A complete paper additionally needs a defensible bridge: independent measures
of the same declared construct before and after randomized exposure, neutral
reference exposure, repeated-measure stability and elicitation-reactivity checks,
and a prespecified delay and attrition analysis. Constructed PAHF labels and a
capable PUPPET reader cannot supply that bridge by themselves. Any human data
route needs realistic access, consent and review arrangements; none is currently
secured. The paper must either obtain an honest external estimand or narrow its
claim and reassess whether the remaining contribution merits a conference paper.

Positioning is deliberately narrow. SLIFT already decomposes feedback, PUMA
models user dynamics, and Privileged Likelihood addresses the gap between
likelihood gradients and utility. Human preference-expression interventions also
have direct prior art. The proposed distinction is the conjunction of a precise
persistence ambiguity, independently defended measurements and a consequential
equal-information learning result. Merely adding a natural-language wrapper or
using a standard estimator does not establish novelty. Exact comparisons and
primary sources are in the [literature audit](AUDIT_THEORY_LITERATURE_20260905.md).

Proceed only through the [bounded completion plan](ICLR_2027_PAPER_COMPLETION_PLAN_20260905.md):
measurement feasibility by September 8; replicated utility, fair baselines and
a complete evidence-grounded draft by September 11–12. The current submission
recommendation is no-go. This pitch defines what a successful paper would need,
not what the evidence already demonstrates.
