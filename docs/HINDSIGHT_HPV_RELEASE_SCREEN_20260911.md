# Additional delayed-outcome release: useful secondary data, not a revival

Decision: no Hindsight training admission. This is an aggregate feasibility
audit of a public release, not a clinical recommendation, neural experiment,
new persuasion intervention or replication of the authors' complete analysis.

The [current v5 paper](https://arxiv.org/abs/2504.20519v5) describes a randomized
trial comparing chatbot conversations, public-health materials and no message,
with immediate, 15-day and 45-day measures. The older search result's 930-person,
15-day framing is not the current release. The [author repository](https://github.com/sehgal-neil/HPV_LLM_Persuasion)
states that transcripts were removed and that follow-up intent is assigned 100
for participants reporting vaccination. Thus the released follow-up variable
combines reported intent and reported behavior, rather than measuring exactly
the same latent preference at every time. The authors provide imputation and
adjusted-analysis code; our calculation below does not replace that analysis.

Pinned public repository revision:
`e90e4165c03384825669ff12a2e425075db52da5`.
Data SHA-256: `11b36b19396d15816183eceebd7a260e79b9df6024e63eb8fa22689bc2fa1743`.
Only aggregate outcomes/arm counts were retained. No transcripts, participant
identifiers, demographic covariates or individual health records were saved,
printed or sent to an external model. Source bytes were read in memory to
compute the checksum and aggregates.

| Released arm | Allocated rows | Immediate observed | Day 15 observed | Day 45 observed |
|---|---:|---:|---:|---:|
| No message | 324 | 290 | 249 | 230 |
| Public-health materials | 325 | 283 | 241 | 226 |
| Default chatbot | 324 | 262 | 225 | 208 |
| Short chatbot | 324 | 261 | 237 | 229 |

All 1,297 records have baseline intent. At day 45, 893 have the released
follow-up outcome and 404 do not. For each arm with N allocated records, sum S
of observed outcomes and M missing outcomes, the finite-cohort mean lies in
[S/N, (S+100M)/N] if missing scores may take any value in [0,100]. Subtracting
control bounds gives the following unadjusted day-45 contrast bounds:

- Default chatbot versus control: [-26.2747, 38.5401] points.
- Short chatbot versus control: [-25.0062, 33.3272] points.
- Public-health materials versus control: [-21.0692, 38.4047] points.

These are elementary missing-outcome bounds for the released finite cohorts,
not confidence intervals, adjusted intention-to-treat estimates or bounds on
individual causal effects. They do not establish a null, negate randomization,
or invalidate an analysis under defensible missingness assumptions. Selection
and sampling uncertainty would require additional treatment. They also do not
distinguish temporary expression, durable intent and actual behavior.

Execution: `scripts/audit_hpv_release_availability.py`; artifact
`artifacts/hpv_release_availability_20260911.json`. An independent pandas replay
using missing-value fills at 0 and 100 verified all 16 arm/outcome summaries.
No GPU calls, imputation fitting, clinical effectiveness inference or model
training occurred. The full registration/clinical analysis was not audited.

This improves the external-data inventory but does not bridge Hindsight's main
gap: there are no released conversations from which to reconstruct the learning
inputs, and the outcome is not an independently validated neutral latent-state
anchor. Do not claim that all possible external data are unavailable; this
specific release is unsuitable for that primary role. Retain it only for a
carefully scoped secondary measured-outcome analysis if a main thesis qualifies.
