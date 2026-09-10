# Cached consensus feasibility: one answer-level witness, no clean reasoning witness

CPU posthoc screen of the existing Qwen3-8B MATH bank; no new model calls.
16 exposed DEV questions, 32 eligible prefixes, eight completions per prefix.
Use only EOS responses with explicit numeric final answers. Unparsed or incomplete
responses are unknown. Strict majority means >=5 of the original8 draws, not
five of a variable parsed subset. This is not an implementation of u-OPSD's rule.
Original manifest and all384 unique sample keys were authenticated.

Only one question/prefix has a wrong numeric majority with a correct numeric
minority: test/geometry/465.json, prefix1. Answers are200 six times,150 once,120
once; all eight terminate and parse. The reference answer is120. These are
answer-level labels, not validated reasoning labels.

Manual inspection of the original question, all eight completion endings, and
the source solution finds a further confound. With theta in degrees, the true
shaded area is pi * (360 + 2 theta)/360. The answer-matching completion instead
uses pi * 5 theta/360, yet both equations equal5pi/3 at theta120. Therefore its
correct final answer does not certify a correct reasoning trajectory. The input
also depends on a diagram supplied as Asymptote text. Both issues preclude
treating this as a clean minority-correct-reasoning experiment.

Decision: no consensus-training launch from this bank. A new study would need
multiple independently checked reasoning witnesses and capability controls,
not merely a non-majority answer matching the final-answer key. This screen
does not show that consensus distillation universally fails, and does not alter
the original fixed-bank stop decision. Raw artifact:
artifacts/pmi_prefix_diagnostic_20260910/CACHED_CONSENSUS_FEASIBILITY.json.

Source-code replay: scripts/audit_math_consensus_feasibility.py. Existing numeric
parser is explicitly a posthoc last-answer heuristic, not a process verifier.
