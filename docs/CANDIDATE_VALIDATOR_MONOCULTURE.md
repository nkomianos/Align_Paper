# Generator--Tester Error Dependence in Agentic Security Repair

## PI decision

Conditionally green-light the bounded G0. This is the current lead because it
combines a practical agent-security failure mode, a controlled crossed
design, an automatic hidden oracle, and a decisive one-GPU test. The broad
claim that cross-family verification can help is already established. The
remaining contribution is narrower: a controlled, within-patch estimate of
generator--tester error dependence for executable security tests, followed by
a held-out test of a behaviorally selected verifier mixture. A positive G0
would not by itself establish an ICLR paper; it would justify the larger
three-family and real-repository study described below.

No paper can be guaranteed acceptance. The purpose of this protocol is to make
the next failure cheap and the next success interpretable.

## Hypothesis

For the frozen Qwen/Gemma pair, when an incorrect security patch and
specification-derived security tests are produced by the same model family,
their errors will be more correlated than when the other family writes the
tests. Under the same fixed proposal budget, the other-family tester should
therefore expose more plausible-but-incomplete security repairs. A DEV-selected
mixture should cover complementary errors and transfer to entirely held-out CWE
families. G0 cannot establish a universal model-family law.

For patch family `i` and verifier family `j`, let `K(i,j)` be the probability
that 12 fixed proposal slots contain a valid test whose input exposes a frozen
vulnerable-baseline defect and on which the candidate retains that defect.
Invalid, duplicate, malformed, and merely functional-regression proposals
consume their slot but cannot count as a primary security kill.
With Qwen and Gemma as the two families, the sole preregistered primary contrast
is:

```text
G = 0.5 * [(K(Qwen,Gemma) - K(Qwen,Qwen))
         + (K(Gemma,Qwen) - K(Gemma,Gemma))]
```

This is stronger than asking whether one model is a better tester. A universally
stronger verifier raises one row and lowers the other in the expression; a
family-specific blind spot produces the same positive direction twice.

## Why this is useful

Coding agents increasingly propose a patch, synthesize tests, and treat those
tests as evidence that the patch is safe. That workflow silently assumes the
verifier is an independent oracle. If generator and verifier share a blind
spot, self-verification can turn a correlated mistake into false confidence.
The actionable design principle is heterologous verification: use a verifier
whose failure distribution is complementary, not merely another sample from
the same family.

## Closest-work boundary

The novelty is narrow and must be stated honestly.

- [When Does Verification Pay Off?](https://arxiv.org/abs/2512.02304)
  already finds that cross-family generative verification often beats self- or
  intra-family verification across 37 models and nine reasoning benchmarks,
  and relates the gain to solution-distribution dissimilarity. It does not use
  executable security tests, retained security defects, or a fixed-within-patch
  factorial design. Therefore generic cross-family benefit is prior work, not
  this paper's claim.
- [Great Models Think Alike](https://openreview.net/forum?id=3Z827FtMNe)
  already shows that model similarity and correlated mistakes undermine LLM
  oversight. This candidate operationalizes a narrower agent-security failure
  with deterministic execution rather than an LLM judge.
- [Refute-or-Promote](https://arxiv.org/abs/2604.19049) already argues for a
  cross-model critic in security defect discovery and reports real cases where
  a different-family reviewer found errors after same-family review. Its
  evidence is a retrospective, evolving, single-operator campaign rather than
  a controlled generator-by-tester estimate; this candidate must not claim the
  first security evidence for heterogeneous review.
- [SWE-Mutation](https://arxiv.org/abs/2605.22175) is the strongest negative
  near-miss. Its 500-instance backbone swap reports less than 1.5-point changes
  and no detectable same-family evaluator advantage. It regenerates different
  mutants for different backbones, however, so it does not identify the
  within-patch interaction; it also studies generic repository mutants with
  patch-visible tests rather than retained security defects under a true
  specification-only arm. Its null result is a serious prior against G0, not a
  result to omit.
- [Calibration-Family Overfit](https://arxiv.org/abs/2607.06596) finds the
  opposite pairing effect for learned sabotage monitors: monitors transfer
  poorly off lineage and perform best on the family represented in their
  calibration data. This makes the direction mechanism-dependent. A full paper
  must compare cold specification-derived test synthesis with calibrated or
  patch-exposed monitoring on the same patch population rather than recommending
  model diversity unconditionally.

- [On the Risk of Coding Before Testing](https://arxiv.org/abs/2607.05139)
  establishes that exposing a tester to faulty code reduces fault detection and
  explicitly leaves multi-model independent workflows open. It does not cross
  patch family by test family.
- [Evaluating and Mitigating the Misguidance Effect of Buggy Code](https://arxiv.org/abs/2607.22883)
  studies implementation-induced oracle bias and specification-only prompting.
  It does not test lineage-specific generator/verifier interactions.
- [The Specification as Quality Gate](https://arxiv.org/abs/2603.25773)
  argues that homogeneous reviewers have correlated errors and reports a small
  planted-bug cross-family panel. It calls its evidence directional rather than
  a controlled demonstration.
- [SecMutBench](https://openreview.net/pdf?id=3IGvziUsqX) supplies security
  mutation operators and shows a large expert-versus-LLM test gap, but does not
  cross the family that produced a patch with the family that tests it.
- [Security Tests as Executable Specifications](https://arxiv.org/abs/2608.09740),
  [SEC-bench](https://arxiv.org/abs/2506.11791), and
  [PATCHEVAL](https://arxiv.org/abs/2511.11019) establish the security-repair
  setting and its hidden-test gap without identifying verifier monoculture.

Accordingly, “LLM tests are weak,” “buggy code misguides a tester,” “models have
correlated errors,” and “different models can help” are not novel claims. The
defensible G0 estimand is the pair-specific generator-by-tester interaction on
the exact same plausible patches. A paper contribution requires all of the
following: a fully crossed family-by-family design, plausible model-generated
security repairs, deterministic hidden security oracles, fixed proposal budgets
plus a matched-valid sensitivity analysis, held-out-CWE transfer, and a
behavioral selection rule fit only on DEV. The term *monoculture* is motivation,
not a result that two model families can establish.

## Frozen G0

### Models

- `Qwen/Qwen3.5-9B`, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, non-thinking text mode.
- `google/gemma-4-12B-it`, revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, non-thinking text mode.

### Corpus and blinding

The developmental corpus has 32 small pure-Python repair tasks: four variants
from each of eight CWE families. Entire CWE families, not prompt variants, are
assigned to DEV or locked TEST. Public records contain the vulnerable function,
signature, and behavioral/security contract. The correct implementation,
hidden cases, and mutant labels are written to a separate private-oracle
artifact that no model-generation process opens. Their transparent generator
definitions remain in the repository for scientific audit, so this is
model-sequestering rather than a claim of secrecy from a human code auditor.
The exact public and private artifacts are hash-committed before generation.

Generated security tests are JSON input/output vectors rather than executable
model-written Python. This makes the runner auditable and prevents a generated
test from reading files, launching processes, or accessing the network.

### Factorial generation

For every task:

1. Each family generates three candidate patches.
2. The hidden oracle partitions inputs *before patch evaluation*. A
   `security-defect` case is an input where the frozen vulnerable implementation
   disagrees with the correct reference; a `functional-retention` case is one it
   already gets right. A patch enters the primary population only if it runs
   within the sandbox, passes every public regression, preserves **every** hidden
   functional-retention case, repairs at least one security-defect case, and
   retains at least one other security-defect case. This rejects hard-coded or
   globally over-restrictive programs rather than treating them as repairs.
3. Every parse-valid patch is normalized with Python's AST unparser before
   either verifier sees it, removing comments, docstrings, markdown, and
   formatting as trivial family fingerprints.
4. In the **primary specification-only arm**, each verifier sees only the
   signature, contract, and public regression vectors—not the vulnerable or
   candidate implementation. It emits two suites of six test vectors per task;
   the same task-level tests are then run against every candidate patch. This
   prevents shared code anchoring from masquerading as a lineage interaction.
5. In the secondary patch-aware arm, each verifier sees the vulnerable source
   and the same AST-canonicalized candidate, then emits two more suites of six
   vectors.
6. Every vector is first run against the hidden correct implementation. Only
   tests that pass it are valid. The same input is then run against the frozen
   vulnerable baseline. It is security-relevant only when that baseline fails.
7. A primary security kill requires all three conditions: the reference passes,
   the vulnerable baseline fails, and the candidate still fails. A valid test
   that catches only candidate-induced under-functionality is retained as a
   secondary behavioral failure and receives no primary kill credit.
8. Each arm exposes exactly 12 namespaced proposal slots. Invalid or duplicate
   output leaves a failed slot rather than excluding the patch. Test vectors
   have frozen JSON depth, byte, container, numeric, and execution-time bounds.
    Candidate executions use a frozen two-second wall timeout; formal verification
    uses the same value and rejects an override. Any timeout is recorded as
    indeterminate and cannot receive primary kill credit.
9. A secondary capped-valid analysis uses the first four unique valid tests in
   stable hash order. Failure to reach four counts as verifier failure, never as
   a reason to drop a patch; reach rates and their family gap are explicit gate
   checks within every held-out CWE. Any timeout, crash, rejected execution, or
   invalid sandbox result makes the scientific decision inconclusive rather than
   contributing evidence for a null.
10. As an absolute-power control, each TEST task contributes the
    lexicographically first planted mutant that independently satisfies the same
    plausible-incomplete eligibility rule. Both specification-only verifiers
    must detect at least 20% of these controls under CWE-macro averaging. In
    addition, **every verifier-by-held-out-CWE cell** must detect at least one
    control and achieve at least a 20% within-cell detection rate. Each frozen
    cell contains four controls, so the minimum attainable passing rate is 25%.
    Complementary blind spots across verifiers therefore cannot average into an
    apparently powered null. Any powerless cell makes the result
    `INCONCLUSIVE_INSUFFICIENT_APPARATUS_POWER`, never a kill.

This is approximately 1,100 short generations. The expected wall time after both
models are cached is 2–4 hours on one GH200; the run script is sequential and
needs only one GPU.

### DEV-selected portfolio

DEV chooses, separately for each patch generator, how many of 12 proposal slots come
from each verifier family. It selects only the allocation policy, never a
TEST-specific test. Ties prefer the most balanced allocation. The allocation is
then frozen and evaluated on the locked CWE families. This is a secondary
mitigation endpoint and must beat the better of both homogeneous verifier
policies, not merely the patch family's potentially weak self-verifier.

### Primary endpoints

The sole primary endpoint is the specification-only, fixed-proposal crossed
same-family penalty `G` above. Prespecified secondary endpoints are the
capped-valid version of `G`, the AST-canonicalized patch-aware interaction, and
the held-out gain of the DEV-selected mixed portfolio over the best homogeneous
policy.

Primary estimates use only common-support tasks that yield at least one
plausible incomplete patch from each generator family; this prevents the two
directions being computed on different task mixtures. Patches are averaged
within task and patch family, tasks within CWE, and CWEs with equal weight.
Intervals resample CWE clusters, not individual patches, tests, or prompt
variants. With only four locked CWE clusters, the reported percentile range is
explicitly a routing heuristic—not a calibrated 95% inferential interval—so every
leave-one-CWE-out estimate must also retain the positive direction and a full
study must add more CWE clusters.

## Decision rule

Expand only if the common-task-support population contains at least 30
plausible incomplete TEST repairs, including at least 10 from each patch family
and four held-out CWE families;
primary `G` is at least 10 percentage points; its CWE-cluster heuristic lower bound
is above zero; each generator-specific direction is at least five points; every
leave-one-CWE-out estimate is positive; the capped-valid interaction retains a
positive direction for both generators; and at least 80% of each verifier's
arms reach four valid tests with no more than a ten-point reach-rate gap.

Kill only if primary `G` is below five points with adequate patch capacity, the
capped-valid reach control passes, and the CWE-cluster heuristic upper bound is
below the ten-point expansion threshold. Values between five and ten points,
wide cluster uncertainty, or a failed validity control are explicitly
inconclusive and do not authorize expansion. The mixed
portfolio is secondary, so there is no uncorrected OR across two primary tests.
Neither expansion nor killing is permitted if an execution is indeterminate or
the planted-control macro or verifier-by-CWE cell-power check fails.

An insufficient number of plausible incomplete security patches is operationally
inconclusive, not a scientific pass. It permits only a preregistered increase in
patch samples; it does not permit changing endpoints or opening TEST to tune the
method.

## What a positive G0 would authorize

The full paper still needs:

1. a third genuinely independent model family;
2. a natural vulnerability-repair slice from SEC-bench or PATCHEVAL;
3. a same-population comparison among cold specification-only test synthesis,
   patch-aware critique, and a calibrated monitor, resolving when heterologous
   versus lineage-matched verification helps;
4. expert-authored tests as an external ceiling;
5. hierarchical estimates across tasks, CWEs, languages, and model families;
6. behavioral error-similarity measures that can outperform vendor/family names
   when routing verifiers;
7. ablations separating lineage from model strength and prompt style; and
8. a frozen external replication by another compute provider or collaborator.

## Main reviewer attacks

- Two families are not enough to establish a general monoculture law.
- Synthetic pure functions may not transfer to repository-scale repairs.
- Conditioning on incorrect patches changes the sampled patch population.
- A hidden suite can itself have coverage gaps.
- Apparent interaction can come from test validity, output length, or one
  universally stronger verifier.

The G0 addresses the last objection and cheaply estimates the first four. It
does not pretend to resolve them.

Operational launch, retrieval, integrity, and termination instructions are in
[`VALIDATOR_MONOCULTURE_G0_RUNBOOK.md`](VALIDATOR_MONOCULTURE_G0_RUNBOOK.md).
