# Human measurement substrate: PUPPET release audit

## Why this changes the next step

The hindsight candidate needs independent pre-interaction measurements, not
another simulation of assumed preferences. A public release now provides a
potential measurement substrate. This establishes data availability only; it
does not validate our binary model or a training correction.

The [PUPPET repository](https://github.com/mitmedialab/llm-manipulation) describes
belief-shift prediction from human conversations. Its existing study already
addresses predicting belief change; we must not present that task as new.
The [paper v1](https://arxiv.org/html/2603.20907v1) reports 1,035 participants.

## Local schema inspection

Pinned repository revision: `999963ac73180178033d78a28f5eb247c20011bf`.
CSV: 8,580,267 bytes, SHA-256
`6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d`.
Local directory: `artifacts/puppet_schema_20260904_v1`.

`scripts/audit_puppet_schema.py` downloaded the public CSV and README into a
fresh directory and emitted column names/nonempty counts only. No participant
text, identifiers, demographics or individual outcomes were sent to any external
model. The CSV remains ignored by Git and must not be redistributed in this repo.

Observed 1,151 rows and 1,151 unique participant IDs. All have status completed;
1,085 have `passed_attention_check=True`, 66 False. Six condition codes have
187–196 rows each. There are 27 unique query IDs, but only one distinct scenario
ID, so that field is not an adequate topic grouping variable.

Columns include pre/post belief ratings, confidence, parsed conversations,
perceived manipulation, trust and subjective benefit/harm. Nonempty strings
are not yet checked for numeric validity, sentinel missingness, or measurement
timing. The release also contains identifying metadata columns: do not use them
as model features, expose their values, or infer participant identities.

## Unresolved before outcome analysis

The release row count does not match the paper's N. Removing attention-check
failures leaves 1,085, not 1,035. Do not invent another exclusion to match N.
The inspected v1 methods/search did not resolve the discrepancy; investigate
version history or published processing code before claiming reproduction.

Only `interface/LICENSE` appeared in the inspected repository tree. Do not
assume a UI-code license automatically licenses the human dataset for training
or redistribution. Data-use terms require clarification before either activity.
Read-only local schema inspection is not an authorization to contact participants
or optimize personalized manipulation. Our purpose is measurement validity and
defensive evaluation, not persuasion effectiveness.

## Candidate analysis, not yet approved as a novel paper

One relevant question is whether downstream conversation text predicts the
*pre-interaction* anchor or instead increasingly tracks the post-interaction
rating. First verify the original paper's exact prediction targets and inputs:
this contrast may already be covered. Any analysis must hold out participants
and preferably query families, exclude survey answers from model input, and
distinguish state measurement from causal attribution.

Even a successful contrast would be observational evidence about what a reader
infers, not proof that SDPO changes preferences or that initial preferences define
welfare. Fixed condition randomization would identify assigned-condition effects,
not arbitrary message-level causal effects. No outcomes have been analyzed yet.

## Other lead

[Commercial Persuasion in AI-Mediated Conversations](https://arxiv.org/html/2604.04263v1)
has a book-selection experiment with post-task satisfaction and debriefing choices.
Its inspected version says data/code will become available upon publication;
no downloadable release was located in this search. It cannot currently serve
as our ready-to-run dataset. No contact or data request was sent.
