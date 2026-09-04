# Single-profile calibration decision — September 4

## Decision

Do not launch the planned 64 training updates on this frozen calibration.
This is an apparatus qualification decision, not a negative result for SDPO,
preference shaping, or the proposed causal paper. No parameters were updated.

The 16-case calibration finished successfully and its complete evidence is local
in `retrieved/sdpo_single_profile_20260904T1045Z`. Receipt verification passed.
The separate configuration forensic audit establishes sampled execution rather
than the intended greedy protocol; the original outputs remain unchanged.

## Evidence and limitations

The preserved assistant-model audit examined 48 responses against their sources.
It is not blinded human ground truth. Its normalized labels contain 17 material
content-error judgments and 14 ambiguous condition judgments. Among six ordinary
responses judged both content-clean and stylistically satisfactory, four retain
both properties under the hindsight teacher. There are zero clearly
style-unsatisfactory, content-clean ordinary responses. Thus the recovery stratum
is empty and the frozen qualification thresholds are not satisfied.

Clear errors include changing an interview into a job offer, reversing who was
concerned about a dangerous job, and changing a prospective four-year separation
into a past four-year separation. Some other judgments are debatable; the audit
retains their reasoning rather than presenting its aggregate as established truth.

A coarse binary style label cannot exclude useful incremental preference gains.
Nor can this small sampled calibration measure the causal preference-shaping
hypothesis. It does show that this particular qualification design cannot support
the planned inference. Do not tune the threshold or rerun seeds until it passes.

## Next prerequisite

Before another learning run, define a prospectively measurable adaptation target
with real baseline headroom and separately test factual preservation. Preserve
positive and negative controls, record effective decoding settings, and distinguish
known-method reproduction from a novel causal experiment. No new GPU run is
authorized by this memo.

Audit files: `research_audits/sdpo_single_profile_assistant_model_audit_20260904.json`,
its normalized counterpart, and `sdpo_single_profile_assistant_qualification_20260904.json`.
Normalization changes field names only; it does not relabel observations.
