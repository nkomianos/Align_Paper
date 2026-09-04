# Stationary format-preference positive-control data

This is apparatus qualification for a full-response interaction-learning
implementation. It does not test preference shaping, causal identification,
human welfare, or a novel personalization algorithm. Successful learning would
establish only that the implementation can learn a deliberately simple stable
preference from truthful response-dependent feedback.

## Frozen data and hidden state

`src/interaction_sprint/sdpo_format_control_data.py` prepares
`artifacts/sdpo_format_control_v2`:64 training,64 evaluation and32 calibration
cases. Eight opaque user IDs recur across all splits. Each user has a fixed
preferred format, with two users each assigned JSON, bullets, Markdown table,
or plain key-value lines. Case IDs and nonce asset/zone facts are disjoint across
splits. Shared user IDs are necessary: an unseen user's randomly assigned
preference is not inferable from an opaque ID. No cold-start claim is made.

Policy prompts provide user identity and two grounded facts, but no format
preference, gold response, feedback, or answer-key field. Private `preference`
and `facts` metadata support the deterministic user/validator only. The runner
must construct model input from the `prompt` field, never serialize the entire
record. Evaluation uses the same core prompt without feedback.

The explicit `calibration_prompt` adds the actual preference to verify model
capability before training. `oracle_feedback` supplies the desired style and
requests correct record values, but does not supply a rendered full answer.
It is allowed only in explicit teacher/calibration input. Training feedback is
produced by `feedback(response, record)` from the response's actual validity;
successful responses receive thanks, while failures receive the relevant
format or factual correction. No fabricated failure or wrong factual feedback
is introduced.

## Predeclared validator semantics

- JSON: exactly two string fields `asset` and `zone`; whitespace/key order and
  optional JSON fences accepted. Duplicate keys, extra keys and trailing prose
  are rejected.
- Bullets: two `key: value` items; -, *, or Unicode bullet accepted; either key
  order, flexible whitespace and blank lines accepted.
- Table: two columns headed Field and Value, a Markdown separator, and the two
  data rows. Optional outside pipes and alignment colons are accepted.
- Plain: two unbulleted `key: value` lines, either order, flexible whitespace
  and blank lines. No prose preamble, JSON, table, or bullet syntax.

Balanced Markdown bold around field labels (including the colon) is accepted
in the three Markdown/plain formats, as are bold table headers. Unbalanced
emphasis is rejected. This does not rewrite JSON literals or alter fact values.
This prospective parser audit occurred before any model generation; v1 data
and its manifest remain preserved, with v2 carrying the new source hash.

Grounding is checked independently by exact key/value bindings. A correctly
grounded response in the wrong format has `content_valid=true` but `joint=false`.
A leading explanation is similarly not mistaken for missing facts, though it
violates these explicitly narrow presentation preferences. Contradictory
duplicate bindings or wrong nonce values fail grounding. This is not a general
semantic hallucination detector. Case-insensitive key parsing does not relax
the exact opaque fact values.

The validator/source is frozen before model outputs. If calibration reveals
poor compliance, record that as an apparatus limitation rather than changing
the parser after seeing results. Preserve raw generations, token caps and EOS
status so truncation is not confused with preference learning.

All seven data/validator tests passed at v2 preparation. The manifest records source
and split hashes. The reference renderer is for CPU tests and any explicitly
declared supervised upper-bound arm only; it must not silently substitute for
on-policy model responses in an SDPO claim.
