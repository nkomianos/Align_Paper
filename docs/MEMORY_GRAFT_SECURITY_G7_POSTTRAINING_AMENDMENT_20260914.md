# G7 posttraining implementation amendment

Date: 2026-09-14

The six source pretraining runs completed under the original frozen G7
protocol. The first posttraining invocation then terminated before any
optimizer construction, optimization step, evaluation, intervention, or
scientific result. It failed while tokenizing the mixed `markers` metadata
object because the runner attempted to send every value to the tokenizer. In
addition to the five registered text fields, that object contains count
dictionaries, token counts, and registered row lists. The tokenizer rejected
the first non-string value with `ValueError: text input must be of type str`.
The failed log has SHA-256
`0f1c9fa5237f00865c66d26d1cf7c766a4e9443b22b9712bf7c5386d79a092cd`.

The amendment changes one operation: the posttraining runner now explicitly
selects `trigger`, `near`, `benign`, `payload`, and `benign_continuation`,
verifies that all five values are strings, and tokenizes only those values. A
regression test supplies the exact mixed metadata shape and verifies that only
the five registered surfaces reach the tokenizer. The amended runner's
canonical SHA-256 is
`9989a0571be8bfc0b02dc57597c68889bf8a51aff38289684e21e520c7683501`;
the amended test file's canonical SHA-256 is
`6ca4e4f92e2891726847da41796165c2ac151fc784dab88d21947f3f1f281f87`.

This amendment changes no model, checkpoint, data, marker text, tokenization,
row address, optimizer, schedule, seed, estimand, threshold, gate, control,
intervention, aggregation, replay rule, or budget rule. No posttraining outcome
was available when it was written. The six already sealed pretraining runs are
retained because their frozen config, preregistration, runner, module, data,
compression, arm, and seed bindings are unchanged. Source posttraining and the
complete source replay will use the amended posttraining runner. The failed
output directory and original receipt remain preserved under their original
hashes.

The original freeze receipt has SHA-256
`2b6febaa76513c49324687219495abb6cba64a1ce3cdc1282a2a7e893aace245`.
An amendment receipt will bind this document, the implementation commit, both
original and amended posttraining-runner hashes, the regression test, and all
unchanged frozen inputs before the repaired posttraining invocation is
launched.
