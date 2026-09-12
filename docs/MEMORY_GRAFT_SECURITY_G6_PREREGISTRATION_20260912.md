# G6 pre-registration: optimizer dose, quality cost, and route distribution

**Status:** frozen before any G6 weight load. G6 is a new confirmatory extension;
it does not alter or rescue G3, G3.1, or G4.

## Motivation and fixed estimands

The existing endpoint evidence uses five training seeds and exhibits near-binary
route choices. A Student-t interval on the mean contrast is a poor description
of that distribution. G6 therefore estimates the probability that an
independent training seed selects each causal route, reports every seed, and
measures the paired clean-quality change caused by poison training. It also
replicates all four AdamW table rates rather than reporting only the extreme.

The route indicator is fixed at a causal ASR contrast greater than 0.15, the
minimum effect used by the downstream deletion assay. This threshold is not a
round-number accuracy gate: a smaller route effect would not establish the
paper's minimum meaningful removal or transfer effect. A route may be called
*typical* only when the lower endpoint of its two-sided 95% Wilson prevalence
interval exceeds 0.5. The 0.5 boundary follows directly from “typical” meaning
that a majority of independently trained models select the route.

Installation eligibility remains 0.234881: 0.15 plus the 1,024-prompt paired
Hoeffding half-width. It is an apparatus reading, not an exclusion rule. A seed
below it is reported as an installation failure and remains in every
prevalence denominator.

## Models, optimizer doses, and seeds

Pythia-410M and Pythia-1.4B use the same immutable model, tokenizer, memory,
trigger, payload, clean adaptation, poison count, and 512-step schedule as the
verified parent experiments. For every seed, the poison blocks and dropout RNG
are paired across table AdamW rates 5e-5 (shared baseline), 1e-3, 1e-2, and
1e-1. The backbone always uses AdamW at 5e-5 with weight decay 0.01. No rate is
selected or hidden after observing results.

Sixteen new seeds per size are fixed in the config. The endpoint distribution
also pools the five already verified, identically configured 410M G3 seeds and
the five fixed-profile 1.4B G3.1 seeds. Pooling is declared after observing the
old results and is used for precision, not described as an independent
confirmation. The new 16-seed panel is always reported separately. With 21
pooled seeds, the worst-case 95% Wilson half-width is 0.1967, below the chosen
precision target of plus or minus 0.20.

## Outcomes

For every seed and rate, report installed attack excess, whole-table necessity
and sufficiency, outside-table sufficiency, final-row necessity and
sufficiency, matched-benign accuracy, near-trigger and untriggered payload
rates, and clean NLL. Report continuous values, medians, and the Wilson
prevalence of effects above 0.15. Also report the smallest rate at which each
seed becomes table-dependent.

Quality is paired within seed against the checkpoint immediately before poison
training. Report post-minus-pre clean NLL and matched-benign accuracy with
10,000-resample percentile bootstrap intervals for the mean and fixed bootstrap
seeds. There is no quality gate and no post-hoc claim of “preserved” quality.

At 410M and table rate 1e-1, repeat the frozen G4 interventions on final rows,
earlier trigger-internal rows, their 72-row union, shared-prefix benign rows,
random rows, and the whole table. Report the per-seed route table and Wilson
prevalence for final-row necessity, earlier-row necessity, history-row
sufficiency, specific history necessity, and benign-control drops. These
categorical readings replace no parent result; they describe the seed mixture.

## Decisions and interpretation boundaries

Each route is decided independently. Whole-table dependence is *typical* at a
size only if its pooled Wilson lower bound exceeds 0.5. Sufficiency, final-row
locality, temporal-history locality, specificity failure, and quality cost have
separate readings and cannot block or rescue that decision. The learning-rate
curve is descriptive unless the paired seed panel shows the same ordered
transition in its registered summaries.

The result may establish that route selection varies among independently
trained models under a fixed architecture and recipe. It may not establish an
unobserved discrete latent mechanism, unpredictability for arbitrary training
runs, or a general property of conditional memory. The 1.4B G3 developmental
selection remains invalid regardless of G6.

## Verification and compute justification

The decisive stage receives full replay: clean adaptation, all four paired
optimizer doses, quality evaluations, component interventions, the 410M
temporal panel, aggregate decisions, and complete prediction rows. The source
manifest is sealed and replay metrics must match within 1e-12.

Existing measured runtimes project at most 14 GPU-hours for source plus replay,
28% of the estimated 40.34 hours remaining. This scale is justified by four
linked deficiencies it resolves with the same paired runs: route-prevalence
precision below plus or minus 0.20, dose-response replication, paired quality
cost, and additional G3/G3.1/G4 endpoint seeds.
