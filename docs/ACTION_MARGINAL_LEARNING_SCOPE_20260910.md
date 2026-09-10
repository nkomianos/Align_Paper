# Action marginalization: corrected scope and staged learning design

The original proposal in the user's a521bd9d attachment concerns policy-gradient
variance and learning efficiency. The forced-letter and whole-expression screens
tested decoding accuracy. Their lack of decoding improvement does not test that
learning claim. The learning proposal is **NOT RUN**, not a valid negative.
The expression runner itself explicitly excluded RL variance claims; the error
is using that limited experiment to dispose of the broader proposal.

The scientific obstacle remains strong: Rao–Blackwellization is established,
and canonical output may remove the entire practical need. Merely verifying the
variance inequality is apparatus, not a new result. No paper is qualified.

## Smallest staged test

Use a single-step typed tool environment with four executable choices and two
real JSON field orders per choice. Parse all alternatives and verify that each
pair produces exactly the same action and reward. Use no conversational history;
this avoids a history confound but limits scope to contextual bandits. A future
multi-step experiment must feed back canonical action histories.

Compare ordinary spelling policy gradients, exact action-marginal gradients,
canonical-only policy gradients, and ordinary policy gradients spending the
extra compute on additional independent environment samples. No entropy or KL
term, clipping, or group-normalized rewards in this initial comparison. Use a
fixed action-independent baseline of 0.5. Include complete sequence probabilities
with EOS and normalize over the declared finite grammar. This is an explicitly
restricted policy, not an unbiased estimator for unrestricted LM generation.

For each context and grammar, compute fixed initial offsets equal to log target
action mass minus initial log-summed action probability. Add the same offset to
every spelling of that action. The target is uniform; this intentionally changes
the pretrained behavioral prior. Freeze offsets after initialization, including
on evaluation contexts, without consulting their rewards. All arms then start
from identical action probabilities. This engineered initialization creates
learning headroom and must not be presented as naturally occurring agent failure.

The first admission stage measures actual adapter-parameter gradient moments by
enumerating the finite action grammar on a small outcome-blind context panel.
It must verify equality of expected gradients, correct normalizer derivatives,
matched initial action masses, exact parsing, and lossless scoring alignment.
Canonical-only has no within-action serialization variance to remove. A reduction
against spelling alone cannot admit a paper claim.

Only after this apparatus stage, freeze a bounded neural learning protocol with
held-out contexts, reset adapters and optimizer for each arm/seed, identical
initial adapters across arms, and at least three independent training seeds.
Do not choose checkpoints after evaluation. Report learning against both executed
environment samples and measured synchronized GPU seconds, with enumeration,
normalization, and sampling costs included. Separate one-time diagnostics from
training but disclose both. Match comparisons using prespecified time checkpoints
and a final common affordable budget; let admitted runs finish.

Stop this direction if canonicalization matches or exceeds action marginalization
at equal compute, if extra ordinary rollouts erase its benefit, or if benefits
require unrealistic alias multiplication. A synthetic learning win only admits
an external structured-tool validation and a current nearest-baseline audit; it
does not establish ICLR novelty. A negative in this restricted organism cannot
disprove all representation-aware policy optimization.

## Implementation and cost status

scripts/action_marginal_policy.py implements matched-mass initialization, exact
normalization, both score estimators, and parameter-space gradient moments.
Its CPU tests compare both gradient means to an independently differentiated
expected reward and check the canonical equality case. These are mathematical
implementation checks, not neural evidence. The full neural learning runner and
its dataset are not yet implemented or admitted. Do not add an executable queue
entry until a dry run establishes memory and seconds per update. Reserve at most
two hours for an initial bounded pilot if measured throughput supports all arms;
otherwise reduce scope prospectively or decline admission. Two hours is a budget
envelope, not a measured runtime estimate or permission for a midrun timeout.

## Frozen neural gradient diagnostic (not yet run)

run_action_gradient_diagnostic.py uses cached Qwen3-8B, only last-layer attention
q/v LoRA rank2 alpha4, seed2026091044, four cardinal-move contexts and two JSON key
orders per action. All offsets are uniform-mass controls. It enumerates spelling
and marginal gradients plus an independently differentiated expected reward;
canonical-only receives the same treatment. These four directions are a tiny
apparatus panel, not four independent natural task families.

Save every per-spelling parameter gradient and its probability, along with input
tokens, sequence scores, offsets and independent reference gradient. The read-only
verifier recomputes means and covariance traces from those saved vectors. Token
alignment is independently tested against a position-dependent toy scorer.

Before any neural output, numerical qualification is fixed at <=2% relative
gradient-mean discrepancy (BF16 backward arithmetic) and <=1e-5 action-mass error
for every context and grammar. CPU double-precision identities are tested at
1e-12. At least20% median within-alias variance reduction admits designing the
learning pilot; smaller reduction stops this restricted organism. This threshold
is a practical screening rule, not a test of novel theory or a population effect.
No gate is relaxed after output inspection. The diagnostic is prepared but must
wait for the running MATH queue. Estimate 5–20 minutes, unbenchmarked; actual
elapsed seconds are logged for each context/grammar. No arbitrary midrun kill.
