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

## Established-parser comparison on the saved outputs

Installed Math-Verify0.9.0 in a separate CPU-only AWS environment and executed
`scripts/audit_math_verify_saved_bank.py` on the hash-checked384-row saved bank.
No new model generation occurred. All384outputs produced a mathematical parse;
301matched the target, versus168original rewards. An independent explicit
box/delimiter numeric check supports300matches. The remaining library match
comes from a non-EOS response whose final extraction is an equation rather than
an explicit final answer. It requires manual review, not automatic acceptance.

Saved `MATH_VERIFY_COMPARISON.json` and `MATH_VERIFY_ANCHOR_CHECK.json` under
`artifacts/gh200_research_20260910`; dependency versions, raw hash, parser settings
and per-row outcomes retained. Different denominators matter: these counts cover
all384rows; the earlier .4140625 metric covered eligible DEV only. Do not compare
them directly as an accuracy delta. Matching a final answer does not certify the
reasoning process. The historical gate remains failed and the32B arm remains unrun.

[Math-Verify](https://github.com/huggingface/Math-Verify) already provides broad
numeric/symbolic extraction. Our restricted transport parser is infrastructure,
not a new mathematical verification method. A future protocol must distinguish
format validity, explicit-answer correctness and EOS rather than use successful
flexible extraction as sufficient proof of an answer.

## Corrected-score feasibility of the proposed follow-up

`scripts/audit_math_gap_feasibility.py` verifies the original manifest and complete
sample population before bounding the first policy's between-prefix gaps. Under
explicit final numeric scoring, only1of16DEV questions has an absolute gap>=.5.
Allowing every non-EOS or unparsed response either correctness outcome raises the
maximum number of potentially qualifying questions to3. The planned comparison
requires4questions with gaps>=.5 in BOTH models. Therefore no32B outcome on these
fixed samples can satisfy that rule under this rescoring and its uncertainty set.

This is a posthoc finite-bank feasibility bound, conditional on the explicit
numeric scoring of completed responses. It is not a population bound, a test of
all policy-dependence hypotheses, or permission to change the historical gate.
Future resampling could change the bound, but requires a new justified protocol.
The bound routine passed exhaustive binary-completion checks. Receipt:
`artifacts/gh200_research_20260910/MATH_GAP_FEASIBILITY.json`.
Consequently the32B follow-up remains closed even after resolving the score
understatement. No additional model generation is useful for this fixed-bank gate.
