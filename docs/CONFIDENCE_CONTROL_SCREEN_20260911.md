# Confidence control screen — 11 September 2026 UTC

Decision: no new GPU campaign admitted. The output-only control below is exact
developmental evidence, not a neural finding or a paper-qualified contribution.
AWS was checked at 02:07:26 UTC: RTX PRO 6000 Blackwell Server Edition, zero
utilization, zero MiB allocated, no compute processes. No job was launched.

## Correction to the adaptive-voting premise

A wrong majority does not establish dependence between samples. Independent
draws can concentrate on a wrong modal answer. Shared question/prefix views
must not be counted as independent questions. In our saved math-bank runner,
the seed is set once before one `generate(num_return_sequences=8)` call; the
eight identical seed fields do not mean eight resets producing identical draws.
This code inspection alone neither proves nor disproves sampling independence.

[ReASC](https://arxiv.org/html/2601.02970v2), Sections 3–4.3, already combines
confidence-based early acceptance with weighted adaptive voting. Its stopping
quantity compares leading answer masses; it is not a direct correctness
certificate. Our inference: demonstrating wrong consensus would not by itself
refute that stated estimand or differentiate a new method. The cached DEV
consensus screen also lacks a clean correct-reasoning minority witness; retain
the limitations in CACHED_CONSENSUS_FEASIBILITY_20260910.md.

## Exact output-only control

Let four real-answer logits be z_i and the abstention logit be a. Change only
the abstention logit to a-s. Define q=softmax(z), r=exp(a-s)/(sum exp(z)+exp(a-s)).
Then the five-option distribution satisfies

    P(abstain)=r
    P(answer i)=(1-r) q_i
    P(answer i | real answer)=q_i
    margin=max_i P(answer i)-P(abstain)=q_max-(q_max+1)r
    d margin / ds=(q_max+1)r(1-r)>0.

Thus increasing s raises the maximum real-option probability and the margin,
lowers abstention probability, and can lower greedy abstention, while leaving
every answer logit difference and the conditional answer distribution fixed.
No knowledge variable, ground-truth label, or new inference is involved. Positive
temperature scaling preserves the same argument with rescaled scores.

This is elementary softmax algebra, not a novel theorem. It establishes a
specificity control: these output signatures alone do not identify a richer
internal mechanism. It does not establish that neural activation steering is
equivalent to an output bias, or refute all evidence for functional confidence.

Implemented and executed `scripts/audit_abstention_output_bias.py` on CPU.
Five strengths on 500 deterministically constructed score vectors produced
2,500 dependent views, not 2,500 real questions or independent observations.

| Strength s | Greedy abstentions / 500 | Mean maximum real-option probability |
|---|---:|---:|
| -2 | 474 | 0.161397 |
| -1 | 354 | 0.260836 |
| 0 | 174 | 0.353324 |
| 1 | 50 | 0.416927 |
| 2 | 1 | 0.450525 |

Conditional-distribution error was at most 3.34e-16. All real-answer rankings
were unchanged. Finite-difference derivative error was below 1.22e-11. Common
logit shifts and remapped abstention positions passed invariance controls.
These counts depend on the constructed score distribution, not a population
estimate. No significance test, model accuracy, or causal mediation percentage
is claimed.

Artifacts: `artifacts/abstention_output_bias_20260911/RESULT.json` and
`CONSTRUCTED_ROWS.jsonl`. The result authenticates the source and raw rows by
SHA-256. Invocation uses a fresh output directory and refuses to overwrite it.

## Primary-source positioning and version boundary

[Kumaran et al., March preprint](https://arxiv.org/html/2603.22161v1), Methods
3.4.3–3.4.5 and Results 4.3, constructs steering vectors using answer–abstention
margins, balances selected options, uses separate steering items, and includes
within-item mediation analysis. These controls must be acknowledged. Our
output-only control does not reproduce its activations, data or mediation fit.

The [September 7 journal version](https://www.nature.com/articles/s42256-026-01293-x)
adds separately elicited verbal confidence and activation decoding. Its primary
page content was available through web search after an earlier direct-page
access failure; the March preprint must not be treated as the final version.
Those additional findings prevent a blanket claim that the entire paper merely
measures competition between output tokens. Full final supplements/code remain
unaudited.

[Reported Confidence in LLMs Tracks Commitment More Than Correctness](https://arxiv.org/abs/2606.29490)
already reports a distinction between commitment and correctness, including
mechanistic interventions. Primary abstract read; full methods not audited.
[RiskEval](https://arxiv.org/abs/2601.07767) already evaluates whether verbal
confidence translates into penalty-sensitive abstention. Primary abstract read;
its broad conclusions are not independently replicated here.

Consequently, neither “confidence can change behavior without improving truth”
nor “models fail to use confidence optimally” is an adequate novelty claim.

## Conditional experiment, not an authorized queue entry

The remaining specific question would be whether a learned internal steering
direction has useful effects beyond a DEV-matched abstention-logit bias and
whether those effects transfer across response interfaces. First inspect the
final paper's complete controls and the commitment paper's methods. Kill this
candidate if they already establish the same contrast; no GPU replication just
to rename it.

If a distinct question remains, freeze one short-token natural QA assay with
disjoint vector-construction, matching and evaluation questions. All output
label permutations remain clustered by base question. Compare no intervention,
learned direction, norm-matched random direction, matched output bias, and an
external calibrated abstention threshold at equal coverage. Fit bias and choose
layer/strength on DEV only. Evaluate both the original and a held-out label or
response interface; save full answer scores and format validity. Primary utility
must compare selective error at matched coverage, not just abstention counts.
Conditional answer probabilities and separate-pass confidence are diagnostics,
not substitutes for that endpoint. Distinguish a mechanistic dissociation from
a useful method improvement.

First complete a format/capability smoke on unique questions. Planning estimate
for a cached 8B short-token pilot: 30–90 GPU minutes, unbenchmarked; source-model
27B replication and downloads are additional and not approved by this estimate.
Measure throughput before fixing the full run. No arbitrary midrun time kill.
Stop expansion on failed interface qualification, absent added effect beyond
the matched baseline, or failure to transfer. Only a qualified useful result
would justify a second family and held-out confirmation. This design remains
unimplemented and unrun; no statistical threshold has been chosen after neural
outcomes because there are no such outcomes.

The ICLR paper remains NO-GO. This turn supplies a concrete negative control and
rejects an overbroad novelty pitch; it does not supply the missing main result.
