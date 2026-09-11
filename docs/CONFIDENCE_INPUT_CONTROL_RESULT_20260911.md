# Confidence input-control smoke: invalid response interface

The prospective protocol at commit fefdc08 failed its first gate. No full
comparison is admitted. No conclusion about the neural hypothesis follows.

On 20 unique MMLU abstract-algebra questions, Qwen3-8B produced zero exact
single-letter EOS-terminated answers under the fixed prompt and decoding.
Seven outputs reached EOS with a different format; 13 hit the eight-token cap.
There were 146 generated tokens. The model usually began an explanation;
one observed EOS output was `D) False, True`. That is not the required single
letter. Do not report the runner's zero `correct` count as zero math accuracy:
the scoring prerequisite failed.

Zero abstention calls, zero feature extractions, zero probe fits and zero
steering interventions were executed. Those endpoints are **not run**.
This is an **invalid interface assay**, with a successfully enforced stop gate.

Reported process duration including model load was 5.664365659 seconds,
approximately 0.00157 GPU-host hours; it is not provider billing or H200-equivalent
compute. The prospective 2–10 minute smoke estimate was conservative. At
02:17:00 UTC the process was absent and AWS GPU utilization/memory were zero.
GH200/OSH were not contacted. No paid instance was terminated.

Raw local root:
`artifacts/confidence_input_control_20260911/confidence_input_smoke_v1`.
Both files in the remote manifest were downloaded and independently SHA-256
verified. All 20 raw rows were counted. Runner SHA-256:
`e64ad7041983f696ff103aace6782da7b9c842283aaf0c2540756eaafaff98ee`.
Input SHA-256:
`eae5cfbce33f6da0229c3f9c493aaec161960c136d342762a8310d1cb9116b3b`.
Remote root: `/home/ubuntu/align_research_20260910/confidence_input_smoke_v1`.
Cached model metadata is recorded by the runner; no new complete weight-file
hash replay was performed this turn, so do not describe this as a fresh weight
integrity certification. Local and deployed source/data hashes matched.

The missing input-availability control remains a source-method question, not
an achieved neural finding. Any future assay must prospectively distinguish
native generation from forced-choice scoring and allocate an appropriate
answer budget. Do not rescue this run by reinterpreting partial explanations,
loosening its parser, or running the 80-question continuation despite failure.
The broader steering campaign remains unqualified; the exact output-bias and
shared-noise controls are preserved at their limited scope. No manuscript
claim or submission status changes.

## One prospective repair: V2 also fails qualification

Protocol a314615 allowed complete responses with explicit terminal markers,
keeping the 19/20 joint-validity, 5–18 correct and 3–17 decision-B gates unchanged.
It used the same exposed 20-question smoke, not new confirmation data.
This repair completed in 318.545279573 seconds (5.31 minutes; 0.0885 host hours).
There were 17,300 generated tokens across 20 answer calls and 19 decision calls.
All 39 calls reached EOS; none hit the new token budgets.

Independent verification authenticated both manifest files, replayed every
token sequence and input prompt using the authenticated tokenizer, and
recomputed terminal parsing and gates. There were 19 valid answer formats,
15 gold-matching answers, 17 valid decision formats and 4 decision-B outputs.
The required 19 jointly valid pairs was missed. The 15 correct count is not
independent validation of the dataset's answer keys or a paper result.

The three exclusions expose two different issues:

- One answer wrapped its otherwise explicit marker in markdown, contrary to
  the frozen output format.
- Two decision responses ended `FINAL_DECISION: C` and `FINAL_DECISION: D`.
  Their text answered the underlying math question instead of choosing the
  A/B return-or-abstain action. This is action-label binding confusion.

Consequently, even an output with an allowed A/B token may need semantic
validation. The four B outputs cannot simply be promoted to reliable abstention
behavior. We do not loosen the parser, replace the action labels after viewing
these results, or infer anything about confidence representations.

Disposition: **invalid interface/role-binding assay**. The larger 80-question
run, feature extraction, probes and steering all remain **not run**. The single
prospective repair is exhausted; no further prompt search is admitted for this
candidate. This avoids spending on a full comparison with unqualified labels.

Raw root: `artifacts/confidence_input_control_20260911/confidence_input_smoke_v2`.
Independent replay: `artifacts/confidence_input_control_20260911/VERIFIED_V2.json`.
Runner SHA: `539c69a3029935d9b9fbb230a47021cacca6eb49b672d32108a367852ddb57a7`.
Verifier: `scripts/verify_confidence_input_control.py`. The fixed-layer scorer
exists but has not run on neural features. This is implementation, not evidence.
At 02:26:23 UTC PID 16623 was absent and AWS utilization/memory were zero.
