# Hindsight previous-answer control — frozen diagnostic

Previous teacher revelation probe: 25/32 correct when explicit feedback opposes
the prior assistant action. Test whether supplying that action changes use of
the same explicit feedback. Reuse all 32 opposite-action cases, not just errors.
This is a mechanistic diagnostic on known cases, not new confirmation data.

Three variants differ only in the line revealing the previous answer:

- Shown: exact original prompt, permitting a numerical replay check.
- Omitted: remove that line entirely.
- Redacted: replace the A/B answer with `[redacted]`, retaining the history frame.

96 CPU forwards, same pinned model/tokenizer, no sampling, training or GPU.
No domain filtering, token truncation, or prompt revision after results. Report
accuracy, target probability, paired recoveries/regressions and option-order
effects. The redacted control is not exactly token-count matched and does not
prove an isolated internal mechanism. Neither removing useful history in general
nor a new SDPO method follows from this small explicit-preference experiment.

Scientific interpretations: improvement without the old answer would implicate
the history presentation in this apparatus. No improvement would redirect the
diagnosis toward preference-following or prompt interpretation. A clean result
would qualify a teacher design for a future learning study, not itself establish
the original endogenous-feedback hypothesis, novelty, or paper acceptance.
