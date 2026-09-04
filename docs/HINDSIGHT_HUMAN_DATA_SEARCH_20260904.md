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

## Latest-version and format follow-up

Checked [PUPPET v5](https://arxiv.org/html/2603.20907v5), August 11, rather than
relying on v1. Section 6 supplies the initial rating to models predicting the
post-conversation rating. Appendix I supplies baseline-belief context to
personalized assistants. Therefore baseline recovery differs from their stated
prediction task, but personalized dialogues have an information channel from the
baseline measurement itself. This must not be interpreted as independent
preference inference. The paper still reports 1,035 interactions.

The repository has only an initial commit and the dataset/interface addition;
no release filtering pipeline was found. Inclusion rules and dataset-use terms
remain unresolved. Do not infer that a paper license covers a separately hosted
dataset, and do not claim the count discrepancy is an error in the paper.

The structural audit now validates all 1,151 numerical pre/post ratings within
0–100, all reported deltas against post-minus-pre, and all duplicate pre-rating
fields against each other. These are consistency checks, not substantive outcome
analysis. All transcripts start with a USER marker; 1,144 end with BOT and 1,143
have alternating role markers. Marker counts match `total_messages` for all
rows (7,155 USER, 7,126 BOT markers). Flat text markers are not authenticated
boundaries, so do not silently repair the eight non-alternating records or drop
them based on a desired analysis outcome.

An initial attempt to parse `conversation_parsed` as a Python literal failed:
the field is flat USER/BOT text, not a serialized list. Its SyntaxError printed
one source BOT line. No participant identifier or user reply was printed, no
code in the dataset executed, and no external model call occurred. The new
`audit_puppet_structure.py` reports only aggregate diagnostics and never echoes
text on parse errors. Receipt: `structure_v1.json` beside the immutable CSV.

## Narrow candidate protocol to review before execution

Test prior-belief reconstruction, not persuasion optimization. Inputs would be
the belief statement plus user-only dialogue at fixed early and late prefixes;
targets would be independently recorded initial ratings, with post-ratings used
only for evaluation. No demographics, condition labels, survey answers, tactics,
incentives, participant IDs, or post-study reflections enter model inputs.

Mandatory controls: query-only prior, initial-user-text baseline, token-budget
matched early/late excerpts, an explicitly time-aware reconstruction prompt,
and separate personalized/non-personalized analysis. Hold out query families and
participants. Compare pre-rating error and post-rating error jointly, controlling
for the trivial difference in target variance and regression to the mean.
Later text moving a prediction toward post-ratings is not by itself evidence of
causal influence or a training failure.

This remains a proposed measurement audit, not a frozen GPU experiment. Resolve
data-use scope, inclusion policy, and close prior work before inference. A
result could inform the hindsight measurement problem; it would not alone prove
an ICLR-worthy paper or validate the binary filter.
