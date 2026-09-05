# Public measured-response audit: executed results

Protocol freeze: commit `e886de5`; statistical source freeze: `84476ab`, before its substantive endpoints. The protocol was written after published results and availability were known; this is a secondary reanalysis, not a preregistered experiment or an independent unseen-data replication. Code,13 reconstruction tests and3 statistical fixtures passed before the corresponding human-data stages. No prediction, subgroup targeting, participant contact or external model call was used.

The hash-pinned current cohort contains1,108 records (568control,540treatment);750 recontacts link uniquely (383control,367treatment). Current and historical initial cohorts were not mixed. Numeric reconstruction reproduced the frozen joins and source-wave mapping, with6,648 participant/wave/scale storage rows. That row count is not sample size. The original files and derived participant scores remain local and excluded from redistribution.

The primary estimand is the change in **recorded arm contrast** between immediate measurement and the observed October recontact round, within this selected analytic cohort. Each missing item takes its full0–100 range. These are exact finite-cohort identification intervals under that range restriction, **not confidence intervals or causal treatment effects**.

| Construct | Immediate arm contrast | Recontact arm contrast | Change in arm contrast |
|---|---|---|---|
| Prejudice, primary | [−6.4384,−5.8302] | [−35.3355,29.9049] | **[−29.5053,36.3433]** |
| Policy support, secondary | [6.2925,6.9428] | [−30.8608,34.8322] | [−37.8035,28.5397] |

Both primary and secondary intervals include zero. The observed cells plus range restrictions do not identify the sign of the change. This is not a negative result for persistence, an equivalence test, or evidence that the source paper's causal conclusion is false under its additional assumptions.

Among723 participants complete on all six prejudice items at both postexposure waves (369control,354treatment), the immediate contrast is−5.0113 and recontact contrast−3.9736, a descriptive change of+1.0377. This selected-responder point estimate cannot replace the full-cohort bounds. No uncertainty interval is shown for this conditional description, and no participant or item was treated as an independent repeated wave.

All49 prespecified missing-item departure settings were reported for each scale. Across the full grid, prejudice change ranges from−17.2350 to21.5444; policy change from−23.2876 to15.7067. The grid is an assumed nonresponse model, not a validated missing-at-random analysis. Its observed-item mean center is not promoted to the primary estimate. Baseline completeness and bounded contrasts are also retained; baseline imbalance is not itself proof of failed randomization.

The recontact timing remains approximately39.86–49.05days (median45.54), with the source's35-day mean unresolved. Postassignment selection limits the population interpretation. Repeated self-report does not identify latent preference state, measurement neutrality, or a bridge to the constructed EndoPAHF objective. Reproducing these response summaries or discovering the availability/timing discrepancy does not by itself supply ICLR-level methodological novelty.

Machine-readable aggregates, full sensitivity grids, item-level aggregate sufficient statistics and source/code hashes: `artifacts/public_response_statistical_audit_20260905_v1/AGGREGATE_RESULTS.json`. Reconstruction availability receipt: `artifacts/public_response_reconstruction_20260905_v1/RECONSTRUCTION_RECEIPT.json`. Only aggregate receipts may enter evidence bundles; never include `LOCAL_ONLY_numeric_scores.jsonl`, raw CSV/RDS or parser-runtime binaries.
