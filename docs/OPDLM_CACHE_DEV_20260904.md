# Modern block-diffusion cache apparatus qualification

## Frozen before model outputs

This is a 16-case local CPU development test on OPDLM-0.6B, not a paper gate.
It tests whether the BERT information-flow apparatus can be transported to an
actual converted diffusion decoder without altering its block attention.
The same exposed WikiText sentence source is used for engineering qualification;
there is no claim of untouched validation or task-level generation quality.

Primary sources:
- [OPDLM paper](https://arxiv.org/html/2606.06712v1).
- [Released implementation](https://github.com/divelab/OPDLM), pinned commit
  `3bb8bf60a882131aedccf43575f896be95eda0f0`.
- [Released checkpoint](https://huggingface.co/divelab/OPDLM-0.6B), pinned revision
  `e6d44e247a86136c9c45e85af5bc6cdbeca73d41`.

The reviewed `sample/bd3lm_rl_rollout.py` supplies the exact model class
definitions and `_prepare_for_sampling` helper. Only those AST nodes execute;
the rollout module's environment changes, workers and task imports do not.
Source hash `2f58c85ddf29d4707d32cecf3d5f3fce60331042d6dfba2bb1e3244dab9b3745`.
Transformers 5.6.2/PyTorch 2.11 CPU replace the author's older runtime; this
qualification must not be described as an exact full-environment reproduction.

Four-token block-causal attention, same-position output alignment. An optional
ARM shift exists in training code but `student_arm_shift: False` in the BD3LM
configuration; the published sampler scores current positions. The separate
SDAR engine's shifted path is not grounds to change this model's alignment.
No prefix KV caching is used in our small full-forward diagnostic.

## Paired computation

For each of the first 16 eligible exposed sentences, tokenize an at-most-64
token prefix ending at a four-token boundary. Mask all four final tokens. Pick
the highest-confidence proposed token among those four as the verification seed.
That selection is based on model confidence, not gold correctness. Build a
candidate-bearing cache with just that seed filled, leaving other three masked.

Compare fresh native forward, instrumented plain, candidate cache inserted at
all layers with exact seed-row correction, and last-layer-only corrected cache.
Score the seed separately from the other three positions. The latter measure
whether candidate context has any drafting benefit; seed equality alone is not
a useful new method. Five forwards per case: **80 planned CPU forwards**.
No training, no paid GPU, no automatic expansion, no guessed runtime guarantee.

Numerical prerequisites: native/plain max logit error <3e-4 and native/late-only
seed max error <3e-4. Tests also require candidate dependence at other positions
on a random tiny model while preserving seed equality and block mask semantics.
Report exact token matches, not semantic correctness. No arbitrary effect-size
cutoff is used for this engineering smoke test. Both benefit and damage count.

Final-layer-only injection has a simple single-seed argument: the first L-1
layers are clean; the final seed query's cache entry is restored; no subsequent
cross-position layer can carry the candidate back. It may provide too little
context to help drafting. Multiple seeds require additional cross-seed analysis:
restoring just each seed's own column is not generally sufficient. This is not
a novelty claim: XLNet/two-stream and shadow verification are required priors.
[Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache) already has layer-aware
refresh for acceleration, but does not establish the same independence property.

Outputs: `artifacts/opdlm_cache_dev_v1`; runner
`src/interaction_sprint/opdlm_cache_dev.py`. All evidence preserved. The verifier
checks source/model/evidence bytes and metric recomputation, not forward replay.

## Completed apparatus result

All 80 forwards completed. The first attempt retained all predictions/logits but
failed to serialize the library's set-valued loading metadata. It remains at
`artifacts/opdlm_cache_dev_v1`, including the runner as it existed then. A
recording-only fix was followed by an identical 80-forward replay in
`artifacts/opdlm_cache_dev_v1_recording_retry`: every saved float32 logit and
prediction matches the first attempt exactly. No first-attempt output was edited.

| Arm | Seed correct /16 | Other positions correct /48 | Changed seed argmax |
| --- | ---: | ---: | ---: |
| Native / instrumented plain | 2 | 1 | 0 |
| All-layer corrected candidate cache | 2 | 2 | 0 |
| Last-layer-only corrected candidate cache | 2 | 1 | 0 |

Native/plain logit error is zero; last-only seed max error 6.91e-6. Numerical
controls pass, but exact continuation reconstruction is poor and the proposed
late-only rule has no observed drafting advantage here. Do not call this a
modern-decoder quality improvement or count 160 forwards as two independent runs.
The correct next step is native task/format qualification, not paper expansion.

Verified retry manifest:
`6627d3609325d394f4e82019d5a44732445d5cf80a6ab03244a502ee92730dc9`.
Weights match the pinned Hub LFS SHA-256:
`b21185037fde14214e7473ac29632770592bcfd6f7f532d73c35ec2a9f5098a5`
(1,192,135,096 bytes). No missing, unexpected or mismatched weight keys. An
AutoTokenizer config-type warning is retained; it is not a weight-loading failure.
Twenty-five combined relevant tests pass after the metadata-serialization test.

## Subsequent native chat qualification

`scripts/qualify_opdlm_native_chat.py` uses the released tokenizer chat template
with thinking disabled, four familiar factual/arithmetic prompts, four-token
blocks and one highest-confidence token filled per step. Max 32 response tokens,
EOS checked at block boundaries, at most 128 full forwards. This is DEV with
known answers, not a benchmark. No cache modifications at all. The first attempt
stopped before inference because Transformers 5.6 returns BatchEncoding by
default; setting `return_dict=False` fixes the caller without altering prompts.
Both attempt directories are preserved. Actual output is recorded separately.

The retry completed **36 native forwards in 16.15 seconds**. All four responses
terminated, with verbatim decoded text:

- `The capital of France is Paris..`
- `Two plus three is five5.`
- `A ripe banana is typically yellow.`
- `Monday, Tuesday, Wednesday.`

These contain the expected facts, but do not satisfy the requested answer-only
format and include duplication artifacts. This supports basic functioning of the
released chat interface, not a four-for-four formal accuracy or reasoning claim.
It makes chat/on-policy states a better next apparatus than unprompted Wikipedia
continuation. Checksums, model/source pins, all four records, termination flags
and step/call counts were checked; not a second replay of these chat forwards.
Chat manifest: `a6f019bc3ab97c45dce2dc36d71379a81fbc8eee3d4d4094f8c70b7ad3eb94ef`.

## PI next step, not an active queue

Freeze a paired audit on actual chat denoising trajectories: select the first
committed seed, reveal subsequent model-produced tokens, and compare independent
fresh remasking with candidate-cache verification at that same state. Measure
both evidence-sensitive revisions and drafting changes; score the completed task
under a fixed parser with its own competence controls. Do not use ground-truth
tokens as though they were model-produced context. Separate the selection policy
from the verifier, retain all trajectories rather than just cases with a gain,
and count complete model calls/cost for the separate-pass baseline.

The immediate purpose is to establish a real decoding failure and whether the
late-only tradeoff is useful, not to assert that a known two-stream idea is new.
If no useful effect appears, stop this correction route. A modern decoder,
mathematical independence and passing instrumentation tests are still not an
ICLR-worthy contribution on their own. No paid GPU resumed; all processes exited.
