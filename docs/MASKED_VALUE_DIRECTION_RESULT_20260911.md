# Masked-value direction audit: no reversal in the specified assay

Classification: valid negative for the specified constructed reversal test;
developmental evidence only. No trained model or paper qualification.

Protocol and executable were committed at a185990 before execution. All 180
settings completed on CPU, enumerating every binomial batch composition rather
than sampling it. Source receipts, probability mass, raw GAE, reward-bearing
positive control, and both masking-invariance controls passed.

**Zero settings produced a negative expected root update.** Across the specified
batch sizes, lambdas, masked lengths and coefficients, the released normalized
direction ranged from +0.01122894044 to +0.35355339042. The true terminal-reward
gradient is +0.5. These magnitudes are not final training utility measurements.

Masked outputs still affect magnitude. At batch size 4, lambda .95, and 16
masked positions, changing the coefficient from -10 to +10 changes the expected
direction from +0.1431859413 to +0.1136392431. Both active-only whitening
(+0.1291345900) and no initial whitening (+0.1300174968) are invariant to the
masked intervention. Thus the null reversal finding does not erase the earlier
mask-dependence finding.

At lambda 1, raw advantages are the negative supplied active values. Final
active-token centering and scaling removes the preceding affine whitening in
exact arithmetic without epsilon. Small residual magnitude differences with
the actual helper's epsilon are not evidence of a directional failure. Final
sample-dependent normalization can still create the positive expected signal
already observed in the earlier calibrated-value audit.

The 180 settings are a fixed diagnostic grid, not independent empirical trials
or a random sample of tasks. This does not establish general safety of the
update, disprove earlier distinct normalization counterexamples, or reproduce
DVPO. Masked outputs are chosen toy values; their natural frequency is unknown.
No PPO clipping, actor optimizer, learned critic, or downstream accuracy was run.

Decision: close this specific follow-up. Do not expand the grid after seeing the
null, train a replacement critic, or spend GPU time on its basis. The broader
paper remains NO-GO; an engineering discrepancy without demonstrated practical
impact and differentiated contribution is insufficient.

Raw report: `artifacts/dvpo_source_audit_20260910/MASKED_VALUE_DIRECTION.json`.
SHA256: `3d6b4efff7b4d00d437233e4a3d728cb133b125afaa6d2764bb8d11d58612f02`.
Runner: `scripts/audit_masked_value_direction.py`. The report records its SHA256
and runtime torch version. No AWS or GH200 connection was made for this audit.
