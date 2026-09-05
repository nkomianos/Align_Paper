# Prospective public measured-response reconstruction and statistical audit

Version 1, 5 September 2026. **Status: protocol and synthetic checks only. No human scale scores, outcome means, treatment contrasts, regressions, correlations or predictive endpoints have been calculated under this protocol.** The separate availability audit already inspected schemas, item definitions, record joins, missingness and timestamps. Published substantive findings have been read. This is a prospective specification of our secondary reanalysis, not an experiment preregistration and not an independent unseen-data replication of the source paper.

**Purpose.** Determine what the obtainable Deep Canvassing release identifies about the change in its recorded arm contrast between immediate measurement and later recontact, under explicit missing-data and selection assumptions. The primary output is an identification interval for the current analytic cohort, not a test of a latent preference state. A reproducible source correction and honest interval can be scientifically useful. Reproducing already published persistence alone does not provide an ICLR contribution or establish an advantage for Hindsight.

This protocol does not optimize persuasion, select recipients, personalize political messages, infer which participants are persuadable, or contact anyone. No dialogue, demographic, political-affiliation or other targeting features enter the reconstruction. It does not authorize external model calls. Reconstruction and statistical analysis are separate recorded stages; only the synthetic tests have run at protocol creation.

## Source binding and population

The primary source is [Costello et al., Deep Canvassing using AI reduces prejudice toward undocumented immigrants, v4](https://osf.io/preprints/osf/q7e6u_v4), its [working release](https://osf.io/syuj4/), [registered snapshot](https://osf.io/c286y/), and the public files below. The snapshot is dated 4 March 2026 and exposes an MIT license; this is not evidence of a prospective experiment registration. Full compliance with the separately claimed AsPredicted #187700 has not been audited.

| Bound source | SHA-256 | Permitted role |
| --- | --- | --- |
| [Current `clean_s1_filt.rds`](https://osf.io/download/698fa83be0dd2d48fbc72a53/) | `1b9a1481df09ff56934ef2c91f619e0159795d6523a81e89d792a2c4513e0dc7` | Fixed current cohort, arm, numeric baseline/immediate items, local join key |
| [Historical recontact CSV](https://osf.io/download/k6ncg/) | `bb521c7c7c5522c1540fffd0a4809ee01cb73275e5cf518e531854b67b40c4b0` | Numeric recontact items and local join key; never replace the current cohort |
| [Initial survey QSF](https://osf.io/download/w56cz/) | `6358123681fb2bc6d46e212039f854871329039e596d844d766860c5b2efc793` | Item definitions and planned assignment flow |
| [Recontact survey QSF](https://osf.io/download/698fa6a446a5b2d579e24c47/) | `4d689fbf3bcafd2f0a1b2e1025ff57f8a4a9d78751df44f6dc301aba660681a0` | Repeated-item identity and export-wave mapping |

The [availability report](HINDSIGHT_PERSISTENCE_MEASUREMENT_FEASIBILITY_20260905.md) binds the source PDF, README, R code and further aggregate receipts. In particular, `01_data_cleaning.R` has SHA-256 `a5ae4f4dbbb05f0834cb90239958c1c7c4bbc9bd0a6420f7d9cce69e141cf2cc`. It calculates composites without changing the original item columns. Its conditional external writing-quality model call must not be executed. The omitted `dc_s1_Combined.rds` can instead be reconstructed from the two bound public data files.

The **primary population is the fixed 1,108 records in the current processed cohort**, comprising 540 treatment and 568 control records. Treatment is `condition=treatment`, `Experimental_Condition=1`; control is respectively `control`, `0`. The QSF planned randomizer assigns immigration versus structurally matched smartphone conversations after baseline blocks. The comparator is an active other-topic conversation, not a validated reporting-only/no-influence intervention.

This cohort can have been selected after assignment: the cleaning code excludes, among other cases, chats that were not started. Therefore all primary arm contrasts are **descriptive contrasts within the selected analytic cohort**. Calling them intention-to-treat effects for everyone randomized requires a separate reconstruction of the original assignment population and selection process. Randomization in the survey design does not remove this conditioning problem.

Do not combine the older 1,310-row initial RDS with the current file or silently add the 122 recontact records outside the current cohort. The 1,108 cohort keys are unique; after the two Qualtrics metadata rows, the historical recontact CSV has 872 unique valid-format keys, of which exactly 750 join. Matched counts must equal 367 treatment and 383 control. No new exclusion based on response content, apparent effect size, writing quality, completion time, disagreement, or researcher intuition is permitted.

## Measurement and timing

The primary scale is the mean of six 0–100 prejudice items, coded so higher values mean more prejudice. The secondary policy scale is the mean of five 0–100 items, coded so higher values mean more pro-immigration policy support. Policy is a distinct construct and cannot replace the primary scale when inconvenient.

| Canonical item | Baseline column | Immediate and recontact column | Transformation |
| --- | --- | --- | --- |
| Prejudice: living | `t1_imm_prej_living_1` | `t2_imm_prej_living_1` | `x` |
| Prejudice: fit | `t1_imm_prej_fit_1` | `t2_imm_prej_fit_1` | `x` |
| Prejudice: burden | `t1_imm_prej_burden_1` | `t2_imm_prej_burden_1` | `x` |
| Prejudice: crime | `t1_imm_prej_crime_1` | `t2_imm_prej_crimes_1` | `x` |
| Prejudice: shared values | `t1_imm_prej_values_1` | `t2_imm_prej_values_1` | `100-x` |
| Prejudice: thermometer | `t1_therm_illegal_imm_1` | `t2_therm_illegal_imm_1` | `100-x` |
| Policy: attorney | `t1_imm_attorney_1` | `t2_imm_attorney_1` | `x` |
| Policy: police | `t1_imm_police_1` | `t2_imm_police_1` | `100-x` |
| Policy: deport all | `t1_imm_deportall_1` | `t2_imm_deportall_1` | `100-x` |
| Policy: DACA | `t1_imm_daca_1` | `t2_imm_daca_1` | `x` |
| Policy: citizenship | `t1_imm_citizenship_1` | `t2_imm_citizenship_1` | `x` |

The historical recontact CSV retains `t2_` tags. **File identity and column identity jointly define the wave.** Immediate and later responses must never overwrite one another in a merge. The extra compassion item is excluded. All six prejudice items have matched baseline/immediate/recontact wording. Immediate/recontact policy items match; two baseline policy wordings differ by a typo/punctuation, as recorded in `ITEM_MAPPING.json`.

The actual matched export dates place baseline sessions on 28 August–6 September 2024 and recontact on 16–19 October 2024. Main-session end to recontact start is approximately 39.86–49.05 days, median 45.54 days. This conflicts with the v4 prose's mean 35 days. The primary target is **the survey's observed October recontact round**, with this heterogeneous elapsed schedule, not a fixed 35-day response. For nonresponders the target means the item responses that would have been recorded under completion of that recontact round; no exact person-specific day is imputed. We cannot estimate a causal time-decay curve from self-selected response timing. A correction that changes source timing or cohort binding requires an amendment before further endpoints, preserving prior receipts.

## Complete-item scoring and missingness

For each wave and scale with `K` items, transform observed numeric values using the table. A **point score is reported only if all K items are observed**. Its value is their arithmetic mean. Do not average whichever items happen to be observed, substitute the authors' available composite, replace missingness with zero, standardize using the test/follow-up distribution, or silently apply psych-package defaults.

Let `s_it` be the sum of observed transformed items and `m_it` the number missing. Define the logically possible score interval

`L_it = s_it / K`, `U_it = (s_it + 100 m_it) / K`.

This retains partial item information while allowing every missing item its entire permitted range. For a nonrespondent the interval is `[0,100]`; for a complete scale the interval is a point. Bounds apply at baseline and immediate measurement too, where item-level missingness exists despite complete stored composites. Values below 0 or above 100, infinities, and nonblank nonnumeric strings are schema failures, not missing values to be coerced. Numeric RDS NaN and blank CSV entries are missing.

The availability audit found 731 complete six-item prejudice scales and 724 complete five-item policy scales among the 750 linked recontacts. It has not counted the joint complete-immediate-and-recontact subset. The reconstruction must emit that count without assuming it equals 731. Partial item missingness and entire-wave nonresponse are reported separately by arm.

## One primary estimand and its exact finite-cohort bounds

Let `A_i` be recorded arm, `S_i=1` indicate inclusion in the fixed processed cohort, and `Y_it*` denote a full K-item recorded-response score, including unobserved entries if completed under the corresponding survey round. Write `C_t` for the difference of finite-cohort means in arm 1 versus arm 0 at wave t. The primary estimand is

`Theta_S = C_recontact - C_immediate`.

This is change in the **recorded arm contrast**. Its sign is not automatically decay: the immediate contrast's direction matters. Do not divide it by the immediate contrast or label an individual change-score regression coefficient the fraction of a causal effect that persists. The individual differences share a baseline and can associate through measurement error.

For each arm a, take means over **all n_a cohort members**, with no complete-case deletion:

`D_a_lower = mean_a(L_i,recontact - U_i,immediate)`

`D_a_upper = mean_a(U_i,recontact - L_i,immediate)`.

The primary interval is

`[D_1_lower - D_0_upper, D_1_upper - D_0_lower]`.

These bounds are sharp under only the item range and observed cells: every missing item can independently attain the endpoint appropriate to its coefficient. No equality of a participant's missing entries across waves or similarity of missing and observed respondents is imposed. Sharpness is mathematical for this unrestricted completion model, not proof of a latent construct or causal treatment effect.

Also show the component intervals for `C_immediate` and `C_recontact` so readers can interpret the primary change. Show the policy analogue as secondary. **Do not treat inclusion of zero as evidence of no persistence or equivalence.** It means the sign of the change is unresolved by these observed cells and the stated range restrictions. If the interval excludes zero, describe its identified direction for this cohort and measurement protocol; the causal and measurement limitations remain.

The population here is the fixed dataset, so these logical intervals are not sampling confidence intervals and need no p-values. Availability alone already implies a width of at least `100*(173/540 + 185/568) = 64.60746` points from missing recontacts; partial missing items and immediate missingness can widen it further. This calculation uses only already inspected counts, not response values. A wide inconclusive interval is a legitimate audit result, not an assay failure to hide or a reason to switch primary endpoints.

## Dependence, selected-responder summaries and sensitivity

The independent record is a participant, not a conversation turn, item, wave or label. All three waves and both scales stay linked within that unit. The 1,108 participants are the finite-cohort denominator; 6,648 participant-wave-scale rows are a storage layout, not sample size. No bootstrap over individual items/turns or treating repeated responses as independent is allowed.

Three explicitly secondary outputs are allowed in the later statistical stage:

1. **Joint-complete-responder description.** Restrict to participants complete on all six prejudice items at both immediate and recontact waves, report arm counts, both arm contrasts, and their difference. This is a selected responder population. Its point estimate is not a full-cohort effect. If uncertainty is shown, use 10,000 participant bootstrap resamples stratified by recorded arm, retaining paired waves, NumPy `PCG64` seed `2026090501`, and two-sided percentile 2.5/97.5% intervals. Label these conditional empirical-responder sampling summaries; they do not repair selection, establish randomization validity, or form confidence sets for latent persistence.
2. **Missing-cell departure grid.** For each scale, arm and wave separately, impute each missing transformed item by the observed mean of that same item plus a fixed offset, clipped to `[0,100]`. Use only the prespecified common arm-specific offsets `delta_control, delta_treatment` in `{-30,-20,-10,0,10,20,30}` points; the same arm's offset applies across items and both postexposure waves. Report the full 7-by-7 primary-contrast grid. If a relevant item has no observations in an arm/wave, skip this grid and retain range bounds. This is a transparent nonresponse sensitivity model, not a validated MAR estimate, multiple imputation, or confidence procedure; its offset-zero center must not replace the primary interval. It assumes structured departures and can miss wave-specific selection, which the primary unrestricted bounds allow. **This grid requires the same hash-pinned numeric source cells to be reopened in P3:** the scale-level reconstruction table deliberately does not retain item identities or item-specific sums. The P3 runner must repeat the source/hash/whitelist/join checks and compute local per-item, arm and wave observed counts/sums there, binding them to its analysis receipt. Do not substitute a scale mean for an item mean or claim the P1 scale table alone reproduces this grid. No such item sums or means have been computed under this protocol.
3. **Baseline measurement description.** Show arm-specific baseline item-completeness counts and bounded scale contrasts, without selecting covariates or exclusions from baseline p-values. A baseline-adjusted causal model is not part of this protocol.

Equal arm retention cannot establish missing at random. A monotone-selection/Lee bound is not included because monotonic effects on response have not been justified. No multiple-comparison significance claim is made from secondary items, scales, offset cells or subgroup tests. There are no demographic or political-affiliation subgroup analyses.

The older release cannot establish all-randomized selection bounds without a separate provenance audit. If an amended audit obtains verified assignment counts `N_a` containing the current cohort, the current cohort arm-mean interval `[L_a,U_a]` can be expanded to `[n_a L_a/N_a, (n_a U_a +100(N_a-n_a))/N_a]` for a wave. This remains a range-bound description of actual assigned records. A causal ITT interpretation additionally requires valid assignment/execution, correct inclusion and interference assumptions. No such extension is currently claimed or executed.

## Reconstruction implementation and release boundary

[reconstruct_deepcanvassing_measured_responses_20260905.py](../scripts/reconstruct_deepcanvassing_measured_responses_20260905.py) pins both public source hashes, checks a literal reviewed protocol hash, requires isolated `pyreadr==0.5.6`, enforces unique joins and expected cohort/arm counts, scores numeric items, and produces a local numeric table plus an aggregate receipt. It has no downloads, external model call, statistical tests or predictor fitting. Its pure interval function is tested on synthetic extremal completions; the CLI does not call it on human records.

The numeric schema is one row per `participant_ordinal`, `wave`, `scale`, containing arm, link-availability flag, number of observed items, complete score if defined, and lower/upper score bounds. There are three waves and two scales per participant. Ordinals reflect source row order and are **linkable**, not anonymized. Dates, actual identifiers, IPs, locations, demographics, political affiliations, original participant text and conversations are absent from this derived schema. The RDS parser necessarily deserializes its original object locally, then immediately selects only the whitelisted numeric/join fields. No original row values are printed.

Both human inputs and derived participant tables remain under the ignored `artifacts/` tree and are excluded from every repository commit, evidence bundle and paper release. The reconstruction CLI accepts only a fresh output directory within that tree and never overwrites an earlier receipt. Public deliverables may include code, protocol, source links/hashes, question definitions and aggregate numeric results. If data redistribution is needed later it requires a separate review; public availability does not make the exports anonymous.

The synthetic suite is [test_deepcanvassing_measured_response_protocol_20260905.py](../tests/test_deepcanvassing_measured_response_protocol_20260905.py). It tests file-wave mapping, reverse coding, complete-item rules, sharp range bounds, malformed values, metadata removal, duplicate/arm failures, preservation of the current cohort, exclusion of unmatched older records, no identifiers in numeric outputs, and attainable interval endpoints. No human source file is loaded by the tests.

## Sequential gates and reporting

| Gate | Required artifact and decision | Failure outcome |
| --- | --- | --- |
| P0: freeze | Hash this protocol, script, tests and synthetic receipt before human scoring. Record that published outcomes and availability were already known. | No retrospective claim of preregistration; changes get dated amendments. |
| P1: reconstruction | Match source hashes, column definitions, 1,108/872/750 cohort/link counts, arm coding, completeness accounting, and known timing discrepancy. Independent review of source-wave mapping and reverse coding. | Stop substantive analysis for schema/provenance mismatch; preserve existing evidence and diagnose without choosing outcomes. |
| P2: scope | Explicitly label selected analytic cohort, observed recontact window, full-range missingness and measured construct. | No causal ITT, fixed-day, latent-state or reactivity claim. This does not block descriptive finite-cohort analysis. |
| P3: statistical audit | In a separately recorded CPU stage, calculate the one primary interval and all prespecified component/secondary outputs; retain script/version hashes and no individual output. | Inconclusive bounds are reported as inconclusive. No primary-outcome switching. |
| P4: scientific use | A useful method contribution must have its own valid estimand, fair cheap controls and independent novelty argument. | Reproduction or an audit correction alone cannot rescue the current ICLR methods submission. |

Estimated work: 0.5–1 focused CPU workday for independent code/protocol review and numeric reconstruction, then 0.5–1 workday for a statistical runner, checks and report. Computation should be minutes on an ordinary CPU; these are planning estimates, not benchmarked runtime. No GPU or human recruitment is needed. If reconstruction is unreconciled by 8 September, remove this route from the current deadline-critical empirical plan. A method without a valid result and independently reviewed novelty by 11–12 September should move to a later venue.

## Optional prediction audit: separate gate, not an active experiment

No predictive split is declared untouched merely because no local prediction has run: the source is public and its published outcomes informed route choice. A later prediction protocol would need a frozen participant-level split before model-dependent outcome inspection, source hashes, a declaration of any test exposure, a label budget and training-only tuning. All waves of a participant must share one partition. Current aggregate descriptive analysis does not make those labels secret.

The only candidate purpose is **measurement/calibration auditing of later recorded scale scores**, not predicting persuadability or choosing actions. A separate protocol must use numeric baseline/immediate item scores only, exclude dialogue and targeting features, and report aggregate error without releasing participant predictions. Required comparators are immediate-score carry-forward, baseline-score carry-forward, training-label mean, and training-fitted linear/ridge or shrinkage calibration under the identical delayed-label budget. Any residual estimator must use independent fitting or cross-fitting and compare against the ordinary delayed-label mean. There is no prospective justification here for a neural or language-model component.

The separately gated protocol must decide before execution whether it targets the observed complete-response population or uses an explicit missingness model. It must lock the metric, participant split, label sampling repetitions and pooled unit of inference, and prohibit selecting a split/label draw because Hindsight wins. Repeated label draws are Monte Carlo repeats over one human dataset, not new independent human samples. Performance on later recorded responses cannot validate the theorem's latent emission model, evaluate an action-caused post-action matching objective, or establish persistent utility of a deployed neural policy.

**Retain** this route for an honest operational measurement audit. **Pivot** empirical wording to later recorded responses. **Kill** the claim that these files alone validate latent persistent preferences, eliminate measurement reactivity, or bridge the restricted theorem to EndoPAHF. All participants received prior questionnaires, with no randomized measurement-on/off arm and no independently calibrated known emission matrix. A stable response shift can still be a stable reporting artifact. These limits survive any favorable statistical result.
