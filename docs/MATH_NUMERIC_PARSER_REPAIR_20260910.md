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

## Replay integration completed

The saved-row verifier now selects the recorded parser, rejects unknown versions
and mismatching parser hashes, and prevents comparisons across parser versions.
Unversioned historical banks use legacy-v1. The dormant original queue explicitly
selects legacy-v1 so its frozen recipe cannot silently inherit a new parser.
It remains stopped after the failed first bank and has not been launched.

24 targeted tests pass, including a complete384-row constructed rational-answer
bank replay. Actual historical MATH replay still yields coverage .5104167,
EOS .9114583, eligible DEV accuracy .4140625 and qualified=false. The additive
receipt is `artifacts/gh200_research_20260910/math_legacy_parser_replay_20260910.json`.
These results verify compatibility and the original classification; they are not
new scientific observations or a paper gate.

## Isolated AWS deployment verified

At approximately07:16 UTC, transferred ten runner/helper/test files from25fcf50
to `/home/ubuntu/align_research_20260910/math_code_25fcf50` on the user-authorized
AWS host. Local/remote tar SHA256 both
`991d3acc5fde9012691ce4e5ff1579ed359050f34b0d7ef02705ca71899f8442`.
All24tests passed in the isolated Python3.12 environment; retrieved log:
`artifacts/aws_research_20260910/math_runtime_25fcf50_tests.log`.
Deployment archive: `artifacts/deployment/math_runtime_25fcf50.tar`.
No model weights, data, legacy queue, credentials or automatic launch were included.
This verifies deployment/import/replay compatibility, not neural throughput.
