# Poison-complexity G0 v4 capability result

## Decision

**Stop. Candidate 2 is closed under the admitted compute budget.** The frozen
3B-versus-7B capability bracket failed its prospective qualification rule, so
the planned fine-tuning factorial was not launched. This is a valid negative
qualification result and a developmental assay failure. It is not evidence for
or against a poison-complexity scaling law.

## Bound execution

- Executed source commit: `6452e7ed34b652b0db22a09143de6883886dd7c5`.
- Runner SHA256: `9c5af089b42191e0b0d54590bd3326d54ca6161936b31e4931b7de1947d4b94d`.
- Config SHA256: `89c6fa9ec676f30d11e433e672ec899e3fb42eb23f733e3cf8641b6523d75b6d`.
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- Models: immutable Qwen2.5-3B-Instruct revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1` and Qwen2.5-7B-Instruct
  revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Qualification thresholds: at least 90% clean default-rule accuracy and at
  least 85% explicit payload-rule accuracy for every model and payload.
- Raw result root: `artifacts/poison_complexity_capability_v4`.
- Top-level report SHA256:
  `e7641440fd8483c9a2f6ce868bebbcd9102855ddaf25d8ab832aa8445834222f`.
- Evidence inventory digest:
  `1aea95ce3864c41821789f2e20485269d470eda7503ed832291c696f11ebded0`.
  This is SHA256 over the lexicographically sorted 13-line inventory formatted
  as `<file SHA256><two spaces><POSIX relative path><newline>`.

The capability-only path loaded base checkpoints and evaluated frozen rows. It
performed no backward pass, optimizer step, or weight update.

## Reconstructed outcomes

Every entry below was independently recomputed from the six raw JSONL files,
128 rows per condition and payload. Raw-generation accuracy and constrained
candidate accuracy agree for all displayed cells. A 3,840-row replay also
matched every regenerated prompt, target, condition and case field; checked
prompt hashes and probability normalization; reconstructed every summary
field; and matched each per-payload summary to the top-level report.

| Model | Payload | Clean | Near trigger | Explicit capability | Gate |
|---|---|---:|---:|---:|---|
| Qwen2.5-3B | constant | 84.38% | 53.91% | 100.00% | fail clean |
| Qwen2.5-3B | projection | 84.38% | 53.91% | 100.00% | fail clean |
| Qwen2.5-3B | conditional select | 84.38% | 53.91% | 90.62% | fail clean |
| Qwen2.5-7B | constant | 98.44% | 100.00% | 100.00% | pass |
| Qwen2.5-7B | projection | 98.44% | 100.00% | 100.00% | pass |
| Qwen2.5-7B | conditional select | 98.44% | 100.00% | 100.00% | pass |

The 3B model can execute each explicitly stated payload rule, including the
conditional selector, but is not robust to the base prompt. Its clean score is
below threshold and the one-character near-trigger changes default-rule
accuracy from 84.38% to 53.91%. A learned trigger comparison against the 7B
model would therefore combine poison learnability with a large pre-existing
instruction and string-sensitivity difference.

Base trigger-condition values are diagnostics only. No triggered behavior was
learned. The 3B conditional model happens to follow the selector on 41.41% of
seen-trigger rows before training, while the 7B model is at 0%; this further
shows that post-training attack-success rates would not share a comparable
baseline without a different estimand.

## Classification and scope

- **Valid positive:** the 7B checkpoint passes all frozen capability and clean
  controls.
- **Valid negative:** the 3B checkpoint fails the prospective clean-control
  gate on all three payloads.
- **Invalid scientific assay:** the proposed 3B-versus-7B factorial cannot
  isolate computational payload complexity from base prompt robustness.
- **Developmental check:** model-size qualification and raw row collection.
- **Never run:** optimizer-memory preflight, all 18 fine-tuning cells, AUC
  comparisons, independent seeds, transfer tests, and any confirmation study.

The prospective v4 protocol states that failure by either model closes the
candidate and admits no further task repair or larger-model search. Running only
the qualified 7B model would remove the capacity comparison; replacing 3B after
observing this result would reopen model selection and consume the remaining
budget on another developmental search. Neither action is warranted for this
paper program.

One reporting defect is preserved in the raw top-level report: failures are
labelled `clean arithmetic`, inherited from v1 even though v4 uses a semantic
default rule. The executed condition, rows, targets, threshold, and values are
correct. The runner label was corrected after retrieval without altering or
rerunning the evidence.
