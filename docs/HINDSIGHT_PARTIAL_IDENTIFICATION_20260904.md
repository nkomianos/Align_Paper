# Bounded-influence identification: constructive population check

## Precise scope

This is an elementary binary sensitivity analysis, not a new-theorem claim,
an SDPO reproduction, or a human experiment. The chosen objective is matching
the initial preference; it is not asserted to be a uniquely correct welfare
definition. Existing dynamic-preference motivation is due to
[Carroll et al.](https://proceedings.mlr.press/v235/carroll24a.html).

Let initial preference Z be Bernoulli(q). Randomized action A is binary and
independent of Z. With action-dependent probability e_a, the subsequent report
copies A; otherwise it reports Z. This combines action-copying expression and
transition: sequential copying rates c_a and r_a give
e_a = 1-(1-c_a)(1-r_a). It does not identify these components separately.
Suppose a valid external upper bound b on both e_a is known.

For exact population message probabilities m_a=P(O=1|A=a),

    m0 = (1-e0)q
    m1 = (1-e1)q + e1.

For b<1 the sharp feasible interval is

    [ max(m0, (m1-b)/(1-b)), min(m1, m0/(1-b)) ].

For b=1 it is [m0,m1]. An empty interval refutes compatibility with this model
and bound. Necessity follows by bounding each multiplier between 1-b and 1.
For sufficiency, every q in the interval has witnesses
e0=1-m0/q and e1=(m1-q)/(1-q), with boundary cases handled by continuity.
Both witnesses lie in [0,b] and reproduce the observed channel. Thus the interval
is sharp *within this restricted model*. General persuasion, heterogeneous
state-dependent copying, action-confounded logging, or longitudinal users are
not covered.

For recommendation probability p, utility is pq+(1-p)(1-q). A proposed update
p->p' has worst-case utility change equal to the smaller endpoint value of
(p'-p)(2q-1). Accepting only nonnegative worst-case changes gives a simple
set-based filter, not a claim to invent robust policy improvement.

## Executed diagnostic

225 designed population cells per bound: q=.1,.2,...,.9 and each action's
copying rate on five equally spaced values from zero to b. Repeated zero-bound
cells are not independent observations. The candidate update sets p' to the
mean report under balanced randomized logging, starting at p=.5. This is a
transparent plug-in learner, **not SDPO or neural training**.

| Bound | Harmful raw updates | Beneficial raw updates retained by filter |
| --- | ---: | ---: |
| 0 | 0 | 200/200 |
| .2 | 0 | 200/200 |
| .5 | 6 | 174/192 |
| .8 | 24 | 126/172 |
| 1 | 44 | 70/144 |

No harmful update is accepted under the correctly specified bounds, as expected
from the inequality—not an independent discovery. The useful diagnostic is that
the filter need not abstain everywhere and preserves benign learning in this
population model, while conservatism grows with uncertainty.

Output: `artifacts/hindsight_partial_identification_20260904_v1.json`.
Four tests include 1,000 randomized feasible channels, attainable endpoints,
boundary cases, and the misspecification counterexample below.

## Critical limitation: compatibility does not validate the bound

True q=.49, e0=0, e1=(.61-.49)/.51 gives observed channel (.49,.61).
An incorrectly asserted b=.2 still yields a nonempty interval whose lower
endpoint is .5125. The filter accepts p=.5->.55 as beneficial, even though true
initial-preference utility decreases. Observations cannot validate that external
bound merely by passing a model-compatibility test.

Therefore do not present this as a deployable safety certificate. Exact channels
also need replacement by simultaneous finite-sample regions before sampled-data
use, and model misspecification requires explicit sensitivity analysis.

## PI next step

This resolves one narrow feasibility concern: partial identification can retain
useful updates rather than universally abstain. It does not yet establish paper
novelty or a language-model benefit. Before any GPU request, evaluate whether
independent sparse pre-interaction anchors can reduce ambiguity more efficiently
than simply learning directly from those same anchors. An anchor-only learner
is the essential comparator; beating unfiltered feedback alone is insufficient.
If the proposed correction only reproduces that baseline or requires an
unverifiable tight influence bound, do not expand it.
