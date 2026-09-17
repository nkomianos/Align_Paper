# G11 frequency-aware write policy: final result

## Decision

G11 is a valid negative for the proposed constructive policy. All 60 registered
cells installed the trigger mapping, the source run and exact replay completed,
and the verifier matched 252 scientific files. Neither inverse cumulative-hit
normalization nor active-row-only updating produced a typical low-cost table
route at either size. At equal stepwise table-update norm, frequency weighting
reduced rather than improved the observed routing prevalence, although the
five-seed paired causal contrasts were not precise enough to meet the registered
impairment threshold.

This result does not weaken G6. G6 established that a sufficiently large table
learning rate can move the causal storage component. G11 shows that the tested
inverse-frequency allocation is not a reliable substitute for that aggressive
policy and does not supply the paper with an ordinary-gradient item-local write
rule.

## Registered design

Each Pythia size used five sealed clean checkpoints, 64 poison exposures, 512
optimizer steps, paired poison examples and dropout RNG, and the same causal
whole-table and target-row interventions as G6. The six fixed policies were:

1. ordinary AdamW at table rate `5e-5`;
2. ordinary AdamW at the known routing reference rate `1e-1`;
3. active-row-only updates at nominal rate `5e-5`;
4. active-row-only updates weighted by inverse cumulative row hits;
5. unweighted active-row updates norm-matched stepwise to the `1e-1` reference;
6. inverse-frequency active-row updates under the same stepwise norm budget.

A seed selected the table route only if installation exceeded 0.234881 and both
whole-table necessity and sufficiency exceeded 0.15. With five seeds, the
registered Wilson-majority definition of a typical policy required 5/5 routes.

## Outcomes

All cells installed at essentially 100% clean-adjusted ASR. The table-route
counts and mean clean perplexity ratios were:

| Model | Policy | Route seeds | Whole-table necessity | Whole-table sufficiency | Clean PPL ratio |
|---|---|---:|---:|---:|---:|
| 410M | ordinary baseline | 0/5 | 0.0002 | 0.0000 | 1.0495 |
| 410M | high-LR reference | 4/5 | 0.8783 | 0.6363 | 1.2527 |
| 410M | active only, nominal | 0/5 | 0.0002 | 0.0000 | 1.0402 |
| 410M | frequency normalized, nominal | 0/5 | 0.0000 | 0.0000 | 1.0568 |
| 410M | active only, norm matched | 4/5 | 0.9920 | 0.7594 | 1.1534 |
| 410M | frequency normalized, norm matched | 2/5 | 0.7994 | 0.3900 | 1.1104 |
| 1.4B | ordinary baseline | 0/5 | 0.0000 | 0.0000 | 1.0608 |
| 1.4B | high-LR reference | 1/5 | 0.1395 | 0.0955 | 1.1045 |
| 1.4B | active only, nominal | 0/5 | 0.0000 | 0.0000 | 1.0608 |
| 1.4B | frequency normalized, nominal | 0/5 | 0.0000 | 0.0000 | 1.0612 |
| 1.4B | active only, norm matched | 1/5 | 0.3139 | 0.1246 | 1.0947 |
| 1.4B | frequency normalized, norm matched | 0/5 | 0.0170 | 0.0004 | 1.0840 |

At nominal rate, the paired frequency-minus-active contrasts are essentially
zero because neither policy routes. Under the matched update norm, the mean
frequency-minus-active whole-table contrasts are -0.1926 necessity and -0.3693
sufficiency at 410M, and -0.2969 and -0.1242 at 1.4B. Every registered interval
crosses the applicable +/-0.15 decision endpoint, so the mechanistic impairment
claim is unresolved rather than passed. The prevalence direction is nonetheless
opposite the constructive hypothesis.

Frequency normalization reduced clean NLL relative to the high-LR reference at
1.4B, by 0.0401 nats at nominal rate and 0.0188 nats under the norm-matched
budget, with both registered lower endpoints above zero. Those cheaper policies
did not route. The 410M cost-reduction intervals cross zero because the
high-rate reference contains a large quality-cost outlier.

Target-row sufficiency is not typical under any policy. Even the active-only
norm-matched arm reaches a mean target-row sufficiency of 0.3213 at 410M but
only 0.0008 at 1.4B, with 4/5 and 1/5 component-route prevalence respectively.
Thus G11 supplies no item-level deletion policy.

## Verification and compute

The source invocation used 7,853.535 seconds (2.1815 GPU-hours); the exact
replay used the same measured duration to retained precision. The verifier
compared 252 files and passed. Source plus replay therefore used approximately
4.3631 GPU-hours. Raw per-seed rows, per-step instrumentation, manifests, and
the verification report are retained on the GPU host; compact decision and row
evidence is mirrored locally under the ignored `artifacts/` tree.

