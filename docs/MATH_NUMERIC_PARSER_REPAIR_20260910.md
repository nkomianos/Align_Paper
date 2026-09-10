# Prospective numeric-answer parser repair

The previous goal turn established the official distillation reproduction cost.
This turn repairs a concrete measurement defect already identified in our math
bank. `scripts/explicit_numeric_answer.py` consumes the entire final delimiter
line and returns exact rational values. It rejects numeric prefixes, malformed
grouping, headings, incomplete fractions, division by zero, and unsupported
symbolic expressions. A malformed final marker cannot silently select an older
valid marker. No arbitrary code or symbolic parser execution is involved.

`run_math_policy_value_bank.py` now defaults to `explicit-numeric-v2`, hashes the
parser in its protocol, canonicalizes targets, and recomputes imported prefix
answer flags using the selected parser. Explicit `--parser legacy-v1` preserves
the old generation/scoring path when historical reproduction requires it.
The historical `run_reasoning_bank.answer` remains unchanged because read-only
verifiers depend on its original semantics. No old result or manifest was edited.

Validation: 22 tests passed across the new transport regressions and existing
reasoning/format tests. Posthoc application to the384 saved MATH bank outputs
parses189 versus196 under the old parser, rejecting seven formerly accepted
strings. Raw source SHA256 and counts are in
`artifacts/gh200_research_20260910/math_explicit_numeric_v2_diagnostic.json`.
This is coverage, not accuracy or scientific confirmation. Boxes, units and
general symbolic equivalence are intentionally outside this numeric protocol.

No rerun is admitted by this repair. The earlier capability/coverage gate remains
failed; repeated use of its DEV examples cannot become independent confirmation.
Future study protocols must specify this transport explicitly and assess EOS,
coverage and semantic correctness independently. No GPU job launched.
