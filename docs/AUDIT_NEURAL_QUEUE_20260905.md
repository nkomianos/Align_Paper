# Independent neural queue audit — 5 September 2026 UTC

Audited checkout: `1cfe22124955dee56c8683cbf0ea0f51137e146a`.
This is a code, mathematics, and existing-data audit. No model was loaded, no
optimizer experiment was launched, and no reserved confirmation record was read.
The old frozen protocols remain historical; the recommendation is **do not launch
G0/G1/G2 unchanged**.

## Findings that change the decision

1. **G1's endpoint is oracle imitation, not persistent-state utility.**
   `src/interaction_sprint/hindsight_neural_policy_g1.py:222` compares absolute
   differences of arm-averaged action probabilities from a trained full-label
   oracle. In the expression population, `P(Z=1)=.75`, so a stochastic policy
   selecting action 1 with probability `p` has correctness `.25+.5p`.
   An oracle endpoint at `.8` has utility `.65`; a candidate at `.99` has utility
   `.745` but receives a worse oracle-distance score `.19`. This is a direct
   mathematical counterexample, not a newly run model result. Closeness to the
   trained oracle cannot establish a policy win for the declared objective.
   Report it as estimator/optimization fidelity or replace the primary utility
   endpoint prospectively, before any neural results.

2. **The option-position guard permits complete per-task failure.**
   `hindsight_neural_policy_g1.py:142` computes the absolute difference between
   two *global* means. Equal and opposite order effects across domains cancel.
   The audit passes a deterministic example to the existing reduction function:
   every domain changes its semantic choice with option order, yet the reported
   position gap is zero. Use paired per-base absolute gaps and disagreement,
   with a fixed tail/max criterion; retain the global number descriptively.

3. **Synthetic G1 does not test new-domain transfer.**
   The 128 training rows contain only 32 distinct prompts: 16 surfaces times two
   option orders. All 16 evaluation surfaces are reused. Removing the introductory
   study/evaluation sentence makes all 32 evaluation prompts match training.
   This is a controlled prefix-transfer test, not independent personalization
   or natural external validation. Row IDs alone do not define independent tasks.
   A population-majority delayed-label baseline and a simple supervised/shrinkage
   baseline are mandatory; the desired action is globally the same in this assay.

4. **Eight panels are not eight independent replications.**
   G0 v2 has 28 unique labeled rows at budget four and 53 at budget eight;
   panels overlap and budgets are nested. The six-of-eight rule is a deterministic
   development rule, not a binomial significance test. G1 does use 64 distinct
   anchors across eight disjoint panels, but shares the finite population,
   surfaces, full-data controls, initialization and schedule. It supplies neither
   eight seeds nor eight population replications. Cluster task variants and
   present seed and sampling variation separately.

5. **The conflict gate mixes scientific signal and interface geometry.**
   G0 requires full-vocabulary raw/oracle cosine at most `-.25`. Its teacher
   qualification permits A/B mass as low as `.20`. Common formatting or vocabulary
   gradients can dominate the cosine even when the two semantic actions favor
   opposite targets. Failure of this contrast is not, by itself, a valid negative
   for the residual estimator. Distinguish interface qualification, existence of
   a semantic conflict, estimator error reduction, and utility. The retained
   original threshold must not be relaxed after outcomes; supersede the entire
   interpretation now if this assay is retained.

6. **The proposed comparison is label-matched, not compute-matched.**
   Each augmented G1 update uses a population student batch plus an anchor batch
   and three teacher batches. Anchor SDPO uses one student/teacher anchor batch;
   SFT uses one student batch. All have 32 updates, but equal update counts are
   not equal FLOPs, forward tokens, or training examples. Give every applicable
   baseline the same accessible immediate logs and delayed labels; report both
   label cost and model compute. Compare against direct anchor means/shrinkage,
   a standard difference/PPI estimator, and action-aware filtering where its
   assumptions are available equally to every method. Do not give a baseline
   privileged latent `z0` or a known corruption law unavailable to the method.

7. **The G1 'power' result is a surrogate grid, not a power estimate.**
   The committed verifier reproduces nine qualifying alternatives and zero of
   nine nulls. These are deterministic transformations of one fixed label sample
   over three gains and three SFT multipliers. The null makes augmented exactly
   equal to anchor SDPO; rejecting it with a strictly positive improvement
   threshold is largely built into the rule. There is no repeated sampling of
   neural studies, uncertainty estimate, or calibrated family-wise false positive
   rate. The 28-cell gradient power replay also passes, but its idealized gradient
   noise model does not validate neural utility or the complete policy rule.

8. **G2 spends compute on an identity and lacks its own early control stop.**
   In `scripts/run_hindsight_pahf_g2_dev.py`, `transition_sanity` subtracts a
   tensor from itself and otherwise repeats the raw arm's deterministic training.
   Agreement of these adapters is an implementation check, not learned
   discrimination of expression and transition. A small unit check suffices.
   All fifteen trained arms run before final control qualification. Evaluate
   raw/full-label acquisition before sparse arms and stop on failure. See the
   separate data audit for cohort dependence, label schedules and power gaps.

## Implementation checks that survive

- The loss is explicitly `KL(student || stopgrad teacher)` over the full
  vocabulary at the **first answer token**. Sign and detach direction are
  correct for that stated loss. It is not an execution of released multi-token
  SDPO, an on-policy rollout objective, or the released top-k-plus-tail recipe.
- The teacher runs with the arm's current adapter under `no_grad`; it changes
  along training. It is detached per step, not a frozen teacher. The method is
  a sequential semi-gradient procedure, so initial-gradient fidelity alone
  does not ensure its later trajectory tracks the oracle.
- `adapter_state` clones tensors to CPU; each arm calls `load_adapter(initial)`
  and constructs a new AdamW with zero weight decay. No cross-arm adapter or
  optimizer reuse was found. The model stays in eval mode, so dropout is disabled
  while gradients remain enabled for the student. Fixed final endpoints avoid
  checkpoint selection; failed arms preserve adapter/optimizer/step evidence.
- The G1 population schedule covers each of 128 records exactly four times and
  keeps the sparse correction separate. Panel builders use row identity and
  randomized action, not latent outcomes. Older `build_records` latent-balanced
  anchor flags are unused by these corrected G0/G1 panel builders.
- G0 and G1 initialize LoRA A with different seeds (`2026090407` and
  `2026090413`). The zero-B base function agrees, but the parameter-gradient
  feature spaces differ. G0 is not qualification of the exact G1 initialization.
  Use an identical serialized initialization for a gradient-to-update claim,
  or explicitly treat G1 as a new independently qualified run.
- G1 launch verifies a qualified G0 before loading weights. Its **offline**
  verifier only checks a saved prerequisite receipt/hash string; it does not
  re-open the actual G0 root. End-to-end re-audit must provide and bind both roots.
- Source hashes bind selected files to the verifier checkout. They do not attest
  the actual weights, whole environment, every imported dependency, or the entire
  Git commit. The launchers check a caller-supplied exact commit but use a broad
  Transformers version floor. Save detached HEAD, effective environment,
  resolved model/tokenizer revisions, weight-file receipts and launch command
  as part of the sealed root before another run.
- G1's offline verifier recomputes saved metrics but does not require the exact
  expected evaluation ID/swap set or verify optimizer hyperparameters/step counts
  inside optimizer states. A manifest verifies included bytes; it does not prove
  those rows were generated from a checkpoint. Harden these checks and separately
  label numerical replay versus neural recomputation.

## Provenance and executed checks

Gradient-v2 protocol commit: `2a15a80` (4 September 13:22:53 PDT).
Current G1 rule/protocol commit: `f68ffdd` (4 September 14:25:13 PDT).
No capable G0/G1 endpoint root was found in the inventory.

Executed read-only verifier commands at the audited checkout:

```
python scripts/verify_hindsight_gradient_power_audit.py --root artifacts/hindsight_gradient_power_audit_20260904_v1
python scripts/verify_hindsight_neural_policy_power_audit.py --root artifacts/hindsight_neural_policy_power_audit_20260904_v2
```

Both returned exit 0 and deterministic replay success (28 cells; 9/9 alternatives,
0/9 nulls respectively). Existing focused tests for neural anchor, gradient,
gradient v2, policy G1 and policy power: **20 passed**. Passing tests do not
remove the endpoint or design defects above.

Independent deterministic data/metric recomputation is retained in
`artifacts/independent_audit_20260905/neural/design_recomputation.json`, generated
by `scripts/audit_neural_design_20260905.py`. This records the exact panel overlap,
exposures, target means, prompt overlap and two mathematical counterexamples.
The audit did not alter any frozen experimental code or threshold.
