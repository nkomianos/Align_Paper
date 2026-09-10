# Conditional paired-prefix replication

Frozen before complete math_bank8_v2 or any math_bank32_v2 outcomes. This plan
does not change the running banks' admission or routing rules. No replication
is yet admitted, launched, or counted as evidence.

Only if BOTH banks qualify and the fixed DEV rule finds at least four large
opposite rankings: select ALL such eligible DEV questions, retain their exact
two prefixes, and freeze the two directions from DEV. Preserve a selection
manifest containing both complete bank hashes. Do not choose the most attractive
four, regenerate prefixes, or alter questions. Public source remains DEV; this
is independent continuation-seed replication of selected cases, not a fresh
task-population confirmation.

Generate32 new continuations per prefix and policy, with the same native prompts,
temperature.8 full-support sampling, disabled thinking mode, and2048-token
finite horizon. Use disjoint random seed blocks across policies and
prefixes, also disjoint from DEV. DEV used the same numerical seed block across
policies; its routing thresholds therefore should not be advertised as tests
based on independent cross-policy samples. The replication seed allocation
removes that avoidable coupling.

At each selected question, run one-sided Fisher exact tests for the TWO directions
frozen from DEV, each comparing32 Bernoulli outcomes against32. Define the
question-level p-value as the maximum of those two p-values: under the union
null that at least one asserted directional effect is absent, this is a valid
intersection-union test. Apply Holm at familywise alpha.05 over ALL selected
questions. This does not require independence across question-level tests.
Conditional on the selected prefixes, the new random draws must be independent
of DEV; pseudo-random generation and finite-horizon parser rewards are the stated
sampling model. This tests the frozen reward endpoint, not latent reasoning value.

A practical confirmed case must pass Holm and have an observed success gap of
at least.25 in each direction. Continue only with at least FOUR such cases. Parser
coverage must be>=.95 and EOS rate>=.90 for each policy; otherwise classify as
invalid capability/format rather than a scientific negative. No sample-size
extension after seeing replication outcomes, no direction changes and no discarded
failures. Count questions as cases and continuations as within-case Monte Carlo
samples. Report all denominators and censored outputs.

Implementation of casewise statistics: scripts/policy_value_replication_statistics.py.
Use this only for new replication outputs. A positive would establish selected
casewise reversals for two capacities within one Qwen family. Policy-dependent
process values already appear in prior work; this is insufficient for a novel
paper or a claim about all PRMs. The next scientific decision must establish a
distinct useful consequence and compare a strong student-conditioned baseline.

Cost is128 continuations per selected question across two policies:512 for four
cases, up to2048 for sixteen. Estimate from final bank throughput before admission,
reserve30 minutes for retrieval, and do not launch if it exceeds the remaining
allocation. Let an admitted run finish; no arbitrary midrun termination.

## Prospective power audit

CPU simulations of20,000 independent trials per scenario retain the frozen rule.
With four selected cases and success probabilities.75/.25 in opposite directions,
the four-case confirmation rate is.9285; at.8/.2 it is.99685. For moderate.65/.35
gaps it is only.03945. Thus a negative is a stop for this large-effect replication,
not evidence that moderate policy dependence is absent. A scenario with a strong
effect in only one policy has probability.0324 of any Holm rejection, and zero
four-case confirmations observed. These are conditional design simulations,
not model results or a proof for all configurations. Exact Monte Carlo binomial
intervals accompany boundary estimates; observed zero is not zero true risk.

Reproduction: scripts/audit_policy_value_replication_power.py. Receipt:
artifacts/gh200_research_20260910/policy_value_replication_power_v2.json. Earlier
v1 contains plug-in Monte Carlo standard errors that become zero at sample rates
zero/one; v2 replaces those uncertainty summaries with binomial intervals, without
changing simulated draws or decision rules. Four targeted statistics tests pass.
