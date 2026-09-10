# Memory external-substrate audit

Classification: developmental data audit, no model experiment.

Pinned `xiaowu0162/longmemeval-cleaned` revision
`98d7416c24c778c2fee6e6f3006e7a073259d48f`; oracle JSON SHA256
`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`.
Files and metadata are in `artifacts/longmemeval_source_20260910`.
`scripts/audit_longmemeval_substrate.py` inventories 500 public evaluation rows
and selects 12 developmental questions by salted ID hash from the 199
non-abstention temporal/update questions, before reading their text or answers.
This is not a locked confirmation set. No full-history benchmark files fetched.

The [official README](https://github.com/xiaowu0162/LongMemEval) describes oracle
retrieval, session timestamps, and evidence flags. Any future prompt must omit
gold answers and evidence flags. Oracle evaluation would test reading under
privileged retrieval, not end-to-end memory. Its existing extraction-and-reasoning
baseline must be included; direct answering alone would be weak.

## Evidence inspected

Read the twelve selected questions/answers and flagged evidence turns for the
first five selected IDs. Three contain successive cumulative counts; another
compares enumerated gym days with a later weekly frequency. These are plausible
natural-language update surfaces, but do not supply gold partial-order graphs.

Question `c6853660` asks whether the latest coffee limit increased or decreased.
Earlier evidence describes a reduction to one cup; later evidence proposes a
two-cup limit and asks for advice. The released answer treats the increase as
settled. My assessment: the text supports an intended change more directly than
an enacted one. This is a specific modality/estimand concern, not proof that the
whole benchmark is invalid. Full unflagged context remains to be checked before
a definitive item-level error claim. No label was changed.

## Consequence

Do not transfer the synthetic memory graph solver onto this dataset by inventing
numeric states, uncertainty labels, or temporal edges. A possible new question is
whether memory systems preserve asserted-versus-planned status during updates,
but that requires novelty review, a specified target and independently supported
annotations. The current audit does not admit a GPU experiment or establish a
paper contribution. The Qwen synthetic positive remains narrow and Nemo's failed
replication remains unchanged.
