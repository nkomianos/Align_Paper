# Conformal reasoning: exact calibration-contract counterexamples

Status: a verified implementation audit, not a paper-qualified new method or a
replication of the authors' benchmark results. This is a distinct benign CPU
follow-up to our structured-memory uncertainty direction. No GPU was used.

Primary sources: [ITCR v1](https://arxiv.org/html/2606.08831v1), particularly
Algorithm 1, Theorem 3.5 and Appendix A.2;
[released calibration implementation](https://github.com/tinattw/ITCR/blob/ccbfcbc80c150fa2f80dabbee4a75b59fd30c2b1/ITCR-main/conformal/conformal.py).
The appendix specifies one earliest-bad score per graph. The code instead pools
all incorrect prefixes. Its no-miss routine calibrates on the last entirely
correct prefix. We tested these code paths, including all three supported
quantile interpolation choices. This description does not establish what code
produced the paper's tables.

## Executed evidence

Runner: `scripts/audit_itcr_calibration_contract.py`.
Source and result: `artifacts/itcr_source_audit_20260910/`.
Commit: `ccbfcbc80c150fa2f80dabbee4a75b59fd30c2b1`.
Calibration file SHA256:
`cb1d8d1e5efbb188c4b15bab3bcc330a1a6645638bf45f899a51c58e34567d84`.
The runner validates the Git blob against the pinned tree, extracts ten reviewed
function definitions with AST, and executes them without importing the external
module. It uses a constant scorer returning probabilities (0.5, 0.5), with twenty
calibration graphs, alpha=0.1, and monotone prefix scores. No external pickle,
trained classifier, or model output was loaded.

Each construction is a degenerate i.i.d. population: every graph has the same
specified labels. Therefore the coverage below is exact for that population;
twenty repeated graphs are not twenty independent scientific replications.

| Population | Released output | Exact target coverage | Diagnostic control |
|---|---|---:|---|
| One false node | Accepts that node at score/threshold 0.5 | No-false: 0 | Strict comparison alone returns empty, coverage 1 |
| Chain of twenty false nodes | Retains 2, 2, or 3 nodes for lower, linear, higher interpolation | No-false: 0 | Strict comparison alone still fails: pooling raises the threshold above the first bad score |
| Chain with labels true, false, true | Retains only the first true node | No-miss: 0 | The required prefix includes all three nodes; the last entirely correct prefix is the wrong calibration target |

All nine released-function checks contradict nominal 0.9 coverage on these
populations. The first case demonstrates a tie problem independent of pooling.
The second isolates pooling from equality handling. The third concerns recall,
which explicitly permits false nodes in order to retain subsequent true nodes.
These examples satisfy score monotonicity; estimating the size penalty is not
the cause here. Assertions passed with torch 2.11.0 CPU, numpy 2.4.2 and
networkx 3.6.1. The result JSON includes runner and source hashes.

## Corrective interpretation

For a fixed expansion path, use one graph-level score at its earliest false
prefix. A conservative lower-tail rule uses k=floor(alpha*(m+1)), the kth
SMALLEST calibration score, and strict acceptance below that score. If k=0,
return an empty graph. Conditional calibration on graphs containing a false
node requires stating that conditioning explicitly. Empty outputs are valid
but potentially useless; efficiency must accompany coverage.

For no-miss, score the first prefix containing every true node, including any
intervening false nodes. Use the appropriate finite-sample upper order statistic
and full-output fallback when that index exceeds the calibration count. Empty
truth sets and score ties need explicit treatment. These are standard conformal
repairs, not a claimed novel contribution. Their formal scope is fixed policies,
fixed score construction and exchangeable graph-level units.

The appendix also appears to reverse the order-statistic direction relative to
its lower-tail objective. The released code uses a lower quantile, so the
appendix discrepancy alone does not diagnose its empirical behavior. Do not
attribute all benchmark conclusions to this textual discrepancy.

## Smallest useful next experiment

Audit the released GSM8K JSON schema, graph validity, question duplicates and
label provenance before fitting anything. Freeze question-grouped development,
training, calibration and evaluation membership before outcome comparisons.
Public author data remain an audit dataset, not our untouched confirmation set.
Compare the released routines with one-score-per-graph, tie-safe repairs using
the SAME fixed scorer and SAME input graphs. Report coverage and retained-node
fraction together, plus nonempty-output coverage descriptively. Separate the
contribution of score quality from calibration correctness. A constant scorer
and simple maximum node-risk scorer are necessary controls.

Estimate: CPU inspection/replay minutes; classifier reconstruction approximately
1-15 CPU minutes after schema validation, currently unbenchmarked. No GPU queue
is admitted by these counterexamples. Stop expansion if the release cannot be
faithfully reconstructed, labels cannot support the target, or repairs merely
recover standard coverage through near-total abstention with no useful new
question. Do not tune on final audit evaluation outcomes. A paper requires an
independently useful result beyond correcting known conformal bookkeeping, plus
independent data and generator-level validation; none is established yet.

## Released-data constant-score replay (completed)

Retrieved the pinned `gsm8k_data.json`, verified its Git blob and SHA256
`a4af2fa804d29c2feb07fdf42940211fc54413be6243a36ce6eec544df02bd62`.
It contains 201 distinct normalized question strings and 1,005 claims. Frozen
question-hash membership: 40 development, 40 training, 60 calibration, 61
evaluation. All adjacency matrices, binary labels and DAG checks passed. This
checks structure, not the semantic correctness of dependency edges or labels.
No fitted classifier or training-set optimization was performed.

Our initial plan incorrectly treated frequency-score as a probability. The
prerequisite check stopped before computing outcomes: 726 values were outside
[0,1], with range -72 to 94. Inspection of the released score-generation code
showed signed support accumulation rather than probability normalization. Thus
the range alone is not proof of malformed data. The available code does not
establish the generation provenance of every stored score. We preserved the
original plan and recorded a pre-outcome amendment that removes the
score-dependent arm, retains all rows/splits, and runs only the constant scorer.
We did not clip, normalize or relabel the stored values to obtain a result.

Runner: `scripts/audit_itcr_released_data.py`. Frozen plan, amendment and result
are in `artifacts/itcr_source_audit_20260910/`. Result:
`CONSTANT_SCORE_DATA_REPLAY.json`. Same constant scorer and size penalty for
released and repaired rules; alpha=0.1 and lower interpolation fixed beforehand.
Released selected prefixes were cross-checked against the original prediction
function bodies. The corrected order-statistic helpers additionally passed
exhaustive tests over discrete i.i.d. populations, including ties, zero through
five calibration samples, and three alpha values. These tests are not a general
proof of the full pipeline.

| Target | Rule | Covered / 61 | Mean retained-node fraction | Nonempty outputs |
|---|---|---:|---:|---:|
| No-false | Released | 56 | 0.4934 | 61 |
| No-false | Repaired | 61 | 0 | 0 |
| No-miss | Released | 60 | 0.9920 | 61 |
| No-miss | Repaired | 60 | 0.9920 | 61 |

Only 15 of 60 calibration graphs contain a false node. For the repaired
no-false arm, the lower-tail order statistic is the minimum earliest-bad score,
which equals the first-node score; strict acceptance returns the empty graph.
Its perfect coverage is therefore vacuous. All reported empirical proportions
meet 0.9 on this one split, but 61 questions cannot establish a distribution-free
guarantee. The synthetic counterexamples survive; an empirical coverage-collapse
claim and a useful repaired-method advantage do not follow from this replay.
The no-miss arms' identical outcome also does not erase their differing contracts.

This completes the smallest constant-score data control. The frequency-based
arm and trained-author-scorer reconstruction remain unrun. Stop automatic
expansion here: standard corrections plus total abstention are not an ICLR
contribution. Further work needs an independently useful question and verified
score provenance, rather than another scorer selected for a favorable result.

## Correction: marginal calibration need not return empty outputs

The preceding empty-output result applies to the conditional-on-error repair,
not to all valid no-false calibration. A standard marginal alternative assigns
one earliest-bad score per calibration graph and +infinity to graphs with no bad
prefix. It targets the requested unconditional coverage without imposing a
stronger conditional guarantee on the error-containing subpopulation.

Executed this additional control after seeing the earlier results, so it is
explicitly posthoc. `scripts/audit_itcr_marginal_control.py` preserves the frozen
splits and constant scorer. Of 60 calibration graphs, 45 have no false prefix.
The sixth-smallest score gives threshold 1.0 with strict acceptance. On the same
61 evaluation questions, it covers 56, retains mean node fraction 0.4934, and
returns nonempty output for all 61: the same observed result as the released
rule. Artifact: `MARGINAL_CONSTANT_CONTROL.json`. Discrete enumeration tests
including the no-error infinity sentinel passed alongside the earlier rank tests.

The elementary rank argument applies to one exchangeable extended-real score
per graph; a false retained prefix requires its finite first-bad score to fall
strictly below the calibrated order statistic. Infinite scores represent no
possible false-prefix event. This is standard conformal calibration, not a new
method. It corrects any interpretation that valid repair necessarily entails
total abstention here. Neither the matched observed outcome nor this posthoc
control retroactively validates the released rule's universal guarantee. There
is still no demonstrated practical advantage or paper-qualified contribution.
