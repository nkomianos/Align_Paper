# Teacher generation diagnostic (frozen before execution)

Question: is the failed full-vocabulary teacher control partly a first-token
format artifact? The source remains the completed, unchanged64 truthful teacher
prompts from `hindsight_full_feedback_cpu_20260904_v1`. Retain every prompt;
no filtering to incorrect or low-mass cases. This is a post-hoc apparatus audit,
not a fresh confirmation set or a change to that run's stopping rule.

Generate greedily from the same pinned Qwen3-0.6B, CPUfloat32, up to24 tokens,
native EOS, thinking disabled. Compare full first-token log probabilities to
the saved teacher matrix; stop on max error>.001. Token IDs must replay exactly.
No training. Save complete outputs, capped/EOS status, source hash and manifest.

Primary diagnostic: entire decoded response after whitespace stripping must be
exactly A or B, correct relative to the same target. Compare with literal first
generated token. Also report a deliberately exploratory unique A/B-mention
parser. It cannot understand negation or rationale and is **not** semantic
correctness. Inspect decoded outputs before interpreting it. Report capped
responses, which may not represent completed answers. No model/prompt search,
automatic expansion, lowered threshold or paper greenlight.

Compute feasibility: only0.6B has cached language-model weights;4B/9B directories
contain metadata. The laptop had about5.1GiB free physical RAM at inspection.
No larger weights downloaded and no cloud instance contacted.

## Completed result

Freeze `b904041`; root `artifacts/hindsight_generation_audit_cpu_20260904_v1`.
All64 prompts completed in63.86seconds,200 generated tokens, zero updates.
Every response reached EOS before the24-token limit. Full first-token log
probabilities replay the frozen teacher within maximum absolute error2.98e-5.
All6 manifest files, source bundle hashes, token/text identities and summary
counts verified with `scripts/verify_hindsight_generation_audit.py`; receipt is
the adjacent `_verified.json`. That audit does not independently rerun the model.

| Diagnostic | Correct /64 |
|---|---:|
| Literal first generated token is the target A/B token |44|
| Entire decoded response is exactly target A/B, whitespace stripped |25|
| Only one standalone A/B label is mentioned, and it matches target |55|

Only27 responses meet the exact whole-response A/B format. The loose parser
recovers11 correct parenthesized-label responses excluded by first-token
scoring. Conversely,19 first-token-correct responses expand their label into
an option description and therefore fail exact whole-response format. No strict
whole-response recovery occurs. Five responses have no standalone A/B label.

All64 decoded outputs were inspected. Examples:

- `(A)` / `(B)`: correct labels, but parenthesis is the first token.
- `A: a circle`: correct label and option; first token passes, strict response
  parser fails due to the explanatory suffix.
- `alpha`, `sans`, `P`: returning an option value or fragment instead of a label.
- `A: Pip` when Pip is option B, and `A: a star` when a star is B: label/value
  contradictions. A value-only judge could mask these errors.

These findings **qualify the earlier teacher diagnosis**:44/64 is first-token
target accuracy, not a semantic preference-understanding score. The teacher is
not simply wrong about preferences on all20 remaining examples. Formatting
accounts for substantial apparent failure, while contradictory or wrong labels
remain. The exploratory55/64 is not a replacement gate metric, independent
confirmation result, or proof of perfect semantic understanding.

The original full-vocabulary96-step learning run and stop remain unchanged:
its objective genuinely copies first-token probabilities, including formatting
mass. It did not establish the intended causal preference-shaping effect.
Future teacher qualification must explicitly separate semantic option choice,
label/value consistency and the action interface used by the learner. Merely
switching to a larger model without checking this would leave the design defect.
Use a declared constrained-choice interface or a sequence-aware objective in a
new prospectively frozen study; do not retrofit either onto these artifacts.

See [the updated novelty boundary](HINDSIGHT_NOVELTY_BOUNDARY_20260904.md): this
apparatus correction is not itself a new paper contribution. No GPU requested
and no follow-up training launched from this diagnostic.
