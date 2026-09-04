# Full-response SDPO positive control: frozen developmental apparatus

This operational protocol follows HINDSIGHT_FULL_SEQUENCE_DESIGN_20260904.
It is a feasibility/positive-control study, not a novel paper or an endogeneity
experiment. No copying-feedback arm runs automatically, regardless of outcome.

## Data and model

Prepared `artifacts/sdpo_format_control_v2`: 64 adaptation episodes, 64 separate
evaluation episodes, 32 calibration episodes. Eight opaque user IDs recur across
splits; two users prefer each of JSON, bullets, table and plain key-value lines.
Each episode contains two fresh nonce facts. Ordinary model prompts expose the
user ID and facts but not the preference. Preferences drive only the structured
user's feedback and evaluation. Hidden preferences cannot be known for a new
user without observations; no unseen-user generalization claim is made.

The checker was frozen before any model generation. It accepts balanced Markdown
bold field labels, reasonable JSON whitespace/key order and a JSON code fence;
it does not rewrite fact values. Nonce facts and case IDs are disjoint. Current
request wording is shared across splits, so this tests new contents for the
same user identities, not unseen renderers or external human dialogue.

Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, native tokenizer,
BF16/SDPA on one GPU. Local tokenizer preflight: all canonical reference outputs
including EOS fit within 39 tokens; ordinary prompts <=79, explicit prompts
<=116, oracle hindsight prompts <=159. Freeze response cap 64 and input cap
1024; no truncation. Dynamic feedback is length-checked before use.

## Method fidelity and departures

The actual `_simple_signal_loss` method is extracted from the pinned
[released updater](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/online_sdpo_updater.py).
Commit `3b17d2a67bd2565b9fbda495fd16a485406aa954`, LF-normalized file SHA-256
`6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d`.
The runtime refuses any other source and checks its value and parameter gradient
against the stopped-advantage formula on independent float64 tensors. The teacher
gradient is absent. Actual model token alignment has a separate tiny-Qwen test.

Full native sampled responses are used, not a restricted answer alphabet.
Training samples at temperature1, top-p1, top-k0. The teacher sees the exact
released hindsight block inserted into the last user message, not a helpful
repair prompt. Teacher and student probabilities are refreshed from the same
current model before each single update. Loss averages token-level negative
stopped-advantage times student log probability, with no signal clipping or
ignored initial tokens.

A fresh GenerationConfig is used for every call; checkpoint-specific repetition,
suppression, forced-token, watermark or other processing settings are not
inherited. Both the unused checkpoint configuration and actual configurations
are archived. Every training trajectory compares generation-time selected-token
log probabilities with the full-prefix base recomputation before its update.
Record the maximum discrepancy and stop before updating if it exceeds0.25 nats.
This is a conservative BF16/CUDA engineering bound, not exact equality of cached
and full-prefix numerical evaluation, and does not hide the measured gaps.

The bridge deliberately retains native generated token IDs and includes EOS
only when actually emitted. Upstream instead re-tokenizes decoded completion
strings and appends EOS. This documented engineering departure prevents a
response/token mismatch; it is not an exact end-to-end released-run reproduction.
Greedy baseline/final evaluation use identical prompts and seed assignments.

Rank16/alpha32 attention q/k/v/o LoRA, zero initial B, no dropout; AdamW lr1e-4,
epsilon1e-6, zero weight decay, norm clip1. These are declared pilot choices.
The released configuration defaults to no LoRA and lr5e-6; we do not claim to
reproduce those hyperparameters. One fixed 64-update schedule is shuffled by
seeded hashes before training, ensuring the original grouped-by-user file does
not impose a blocked continual-learning curriculum. Seed9047801. No sweeps,
checkpoint selection or extra SFT. All 16-update adapters and the final optimizer
are retained. Model eval mode disables stochastic layers while gradients remain
enabled; this Qwen checkpoint has zero dropout.

## Qualification and decisions

Before adaptation, generate three calibration views: ordinary prompt, explicitly
stated preference, and ordinary prompt plus the actual frozen user feedback from
its first response in the exact hindsight block. Require explicit joint
format/content accuracy >=90%, hindsight content accuracy >=90%, and >=75% joint
recovery among initial mismatches. If there are no initial mismatches, recovery
is undefined, not zero. Counts and full outputs are retained.

Failure stops before training as `UNQUALIFIED_GENERATIVE_APPARATUS`. It is not
a scientific rejection of SDPO or preference shaping. After qualification,
ordinary calibration joint accuracy must be <=85% to leave sufficient adaptation
headroom. Higher accuracy stops as `INSUFFICIENT_ADAPTATION_HEADROOM`; no records
are filtered or regenerated. No evaluation-based checkpoint or optimizer choice
is permitted.

Run precisely 64 truthful-feedback updates. At the fixed final checkpoint report
all user-level and episode-level results, preferred-format adherence, content
validity and joint success. The planning positive-control criterion is joint
success gain >=10 points and content loss no worse than5 points. Report raw counts
and paired changes; crossing this cutoff by one case is not an acceptance claim.
An improvement only from explicit prompting does not satisfy this criterion.

Only a verified positive learning result can motivate a separately frozen
expression-endogeneity comparison. It still would not show persistent preference
transition, and there is not yet a novel correction. Do not run an endogenous arm
if this positive control fails.

## Evidence and execution

Save native input/output token IDs, decoded text, sampling token log probabilities,
teacher/student training log probabilities, feedback, per-token-derived loss,
gradient norms, timings, checks, model-file hashes, sources and all checkpoints.
Undefined one-token sample-standard-deviation metadata from the released loss is
stored as null, not zero; objective nonfiniteness remains fatal. A terminal status
and manifest distinguish scientific prerequisites from hardware errors.

Example (root owns staging and launch; use a fresh directory):

```sh
python -m interaction_sprint.sdpo_format_positive_control \
  /path/to/sdpo_format_control_v2 /fresh/sdpo_format_positive_control_root \
  --model-path /home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --upstream-file /path/to/pinned_online_sdpo_updater.py --threads 4
```

There are at most 288 short completions:96 calibration,64 baseline,64 training,
64 final. The two training forward passes and backward are additional. A 2–4 hour
allocation is a conservative planning envelope, not measured throughput; actual
timing and remaining rental window govern launch. Stop only at saved boundaries
if externally instructed, preserve partial work, and reserve retrieval time.
