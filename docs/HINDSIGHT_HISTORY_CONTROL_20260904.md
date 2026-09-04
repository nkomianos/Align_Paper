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

## Result: history removal is not a general fix

Completed 96 forwards in 51.07 seconds on CPU, zero updates. All 32 shown-history
probability vectors replay the previous run exactly (maximum difference zero).
Six manifest files and paired prompt construction verify. Minimum A/B mass is
0.9999954. Receipt:
`artifacts/hindsight_history_control_cpu_20260904_v1_verified.json`.

| Variant | Correct / 32 | Mean correct-option probability | Recoveries / regressions vs shown |
| --- | ---: | ---: | ---: |
| Shown | 25 | .67927 | 0 / 0 |
| Omitted | 16 | .57227 | 1 / 10 |
| Redacted | 30 | .89614 | 5 / 0 |

The omitted variant chooses A in every case: 16/16 correct with target A and
0/16 with target B. Shown scores 15/16 and 10/16; redacted 16/16 and 14/16.
Thus simply deleting history makes this small-model apparatus worse, while
retaining its framing and redacting the old answer improves it. This is a
prompt-sensitive result, not evidence that generic cleanup always improves
reasoning, nor an identified internal anchoring mechanism. There are only eight
semantic domains, reused from the previous diagnostic.

## Relation to the actual Hindsight proposal

The [original SDPO paper's Table 1](https://arxiv.org/html/2603.12273) appends
future user feedback to the prior user context and scores the assistant response
as a completion. Our explicit previous-answer line is a diagnostic addition,
**not a demonstrated bug in that published template**. Neither the shown nor
redacted variant is a faithful full-SDPO reproduction. Any next learning test
must include the actual published construction and a plainly stated-preference
oracle instead of assuming this diagnostic wrapper is the right teacher.

Generic self-contamination and history-cleaned distillation are already covered
by [MAIGO](https://arxiv.org/abs/2605.27186). Role-dependent correction is also
studied by [The Self-Correction Illusion](https://arxiv.org/html/2606.05976v1).
The current plain-text redaction is not the same intervention as changing native
chat roles, but these findings rule out claiming broad cleanup novelty from our
tiny result. The purpose here is to improve measurement of the original
preference-feedback question, not to rename a known effect as a new paper.

PI decision: retain redaction as one apparatus candidate; do not train from it
without fresh paired preference and published-template controls. Do not kill the
original Hindsight idea on this result. No GPU or expansion is running.
