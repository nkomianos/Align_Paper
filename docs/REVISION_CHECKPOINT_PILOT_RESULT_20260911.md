# Revision checkpoint pilot: interface failure, no observed answer-level deficit

**No paper signal and no follow-up run admitted.** The fixed 128-output AWS
pilot finished normally. Its strict output-format assay failed, while a separate
retrospective reading of explicit terminal answers finds every answer correct
for both models. This is not evidence of general capability loss, revision
resistance, or a qualified new method.

| Measurement | Base | Released adapter |
|---|---:|---:|
| Completions reaching EOS |64/64|64/64|
| Frozen plain ANSWER-line format accepted |6/64|10/64|
| Retrospectively interpreted final answer correct |64/64|64/64|
| Generated tokens |117,027|151,336|
| Longest completion |4,544|5,299|

## Evidence and independent verification

PID 12065 terminated and the GPU process list was empty after completion.
All raw files were retrieved to
`artifacts/revision_checkpoint_pilot_20260911/revision_checkpoint_pilot_v1`.
Original manifest SHA256:
`20f68c22acda694eb27c825d2e665c7b6ace25b1eaa0d38ccbcd7424aa0e5b9a`.

The original scorer completed; the independent verifier authenticated every
manifest member, reconstructed gold counts from saved question text using
enumeration, checked output coverage, decoded every token sequence with the
authenticated tokenizer and checked EOS. Its frozen route is
`STOP_INADEQUATE_INTERFACE_OR_CAPABILITY`. In this run the strict interface is
the demonstrated problem; the route name must not be paraphrased as proof that
the models lacked counting capability. Both scorers' original metrics remain
preserved in SCORED.json and VERIFIED.json.

## Why the apparent errors are not mathematical failures

Inspection found bold ANSWER lines and terminal boxed integers. The strict
reader rejected these presentations even when their numeric answers were right.
The separate posthoc reader accepts only explicit terminal answer forms; it does
not search for an arbitrary occurrence of the gold number inside the reasoning.
Its initial two rules resolve 63/64 outputs per model. The remaining two outputs
use a bold label followed by a plain integer, **ANSWER:** 495 and **ANSWER:** 42.
A third explicit rule resolves these. Both diagnosis versions are preserved as
FORMAT_DIAGNOSIS.json and FORMAT_DIAGNOSIS_V2.json.

Under that retrospective reading every history/update cell is 8/8 for both
models. The unique-case count is seven, not eight: bases 4 and 7 duplicate the
entire input family. Deduplication does not create an answer-level difference.
The unchanged old/current-anchor arms also intentionally contain identical
histories. These are repeated controls, not independent replications.

This reader was chosen after seeing output formats. It does not retroactively
pass the frozen interface gate or provide independent confirmation. It does
prevent incorrectly concluding that the models failed their mathematical tasks.
Only final-answer correctness was checked; the reasoning traces were not fully
verified. A small ceiling-level pilot cannot exclude revision problems elsewhere.

## Additional limitations and resource accounting

Generation uses greedy thinking-mode inference, contrary to the model card's
recommended sampled setup. The native template alone does not make this a
recommended-decoding replication. No output hit the 8192-token cap, and no
alternate EOS token 151643 occurred, so those two concerns did not censor this
run's observed completions. The unknown historical base revision still prevents
claiming exact reproduction of the adapter's original evaluation.

The experiment used constructed assistant histories, an explicit instruction to
prioritize latest parameters, seven unique parameter cases, two counting
families and one adapter. It does not measure natural conversation robustness,
general post-training effects, or population-level equivalence.

Inference took 2,481.6824 seconds (41.36 minutes); total program time was
2,497.7036 seconds (41.63 minutes). The planning estimate was too optimistic.
These are program timers, not provider billing or H200-equivalent hours. The
unmerged LoRA implementation and different response lengths preclude interpreting
wall-clock differences as intrinsic reasoning efficiency.

Decision: stop this specific pilot lead without a prompt or sampling sweep.
Any future experiment must check intended decoding, real output formats and
input uniqueness before allocating a full run. This corrects our apparatus;
it supplies neither a model-failure claim nor an ICLR submission contribution.
