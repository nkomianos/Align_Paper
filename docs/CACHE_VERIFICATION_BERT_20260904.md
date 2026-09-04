# Cached verification on a pretrained masked LM

## Decision

**The independence failure survives a pretrained network, but a useful
error-correction method or paper is not established.** Do not request a GPU
expansion on these results alone. This was an independent BERT port, not an
execution of the released COVER diffusion decoder.

Completed: 24 frozen handwritten cloze cases, ten scoring conditions plus two
cache-building forwards per case, **288 CPU forwards, zero training updates**.
The measured inference loop took 22.41 seconds; model download/loading and
verification are outside that timing. All evidence is local and byte-verified.

## Upstream source audit: new evidence

The previous turn did not locate the official implementation. This search did:
[xyzCS/COVER](https://github.com/xyzCS/COVER/tree/188a341f5a1a0da9b08af2f13448d3f34b147486),
cloned read-only at commit `188a341f5a1a0da9b08af2f13448d3f34b147486` into
`artifacts/cover_source_audit_20260904`.

The inspected LLaDA-instruct path saves current seed KV, overrides seed columns,
corrects their diagonal contribution, and passes one shared hidden-state tensor
into the next transformer block. Its scheduler selects previously unmasked
positions, excluding the current verification seeds, then saves their KV for
later verification. Thus our concern is not based on assuming the cache always
contains a freshly predicted token that was actually masked at cache creation.
This source inspection still does not measure behavior of the full scheduler.

Relevant pinned source:

- [modeling_llada_kv_cover.py](https://github.com/xyzCS/COVER/blob/188a341f5a1a0da9b08af2f13448d3f34b147486/llada_ins/code/dllm_eval/models/cover/modeling_llada_kv_cover.py),
  correction at 138–223, override at 1138–1145, block composition at 1863.
  Local byte SHA-256 `b9bdff4b4fd93136e4046f4bef5e92e009afee9ea18f51af2685b3daea9ab4ca`.
- [contextual_cover.py](https://github.com/xyzCS/COVER/blob/188a341f5a1a0da9b08af2f13448d3f34b147486/llada_ins/code/dllm_eval/models/cover/contextual_cover.py),
  eligibility at 540–549 and cache capture at 599–628.
  Local byte SHA-256 `66d00dd8a55fa8a5620231d09bfd68b6d3eeb329c4cb61f90a4a85b063e7cbb2`.

An optional local unit test extracts only the inspected, hash-pinned standalone
correction function via Python AST. Its arithmetic matches direct attention-row
recomputation in float64. No upstream loader, evaluator, launcher, generated code,
or complete model was executed. The upstream checkout remains clean.

## Frozen apparatus

Model: [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594),
revision `86b5e0934494bd15c9632b12f734a8a67f723594`. Public safetensors weights,
CPU float32, eager attention, evaluation mode, two CPU threads. BERT is a cheap
test of pretrained bidirectional information flow, **not a current diffusion LM**.

Cases were written and frozen before inference in
`configs/cache_verification_cloze_v1.json` (SHA-256
`562c055487855000569aa045ef3c1fdd091aa3aecd0b03147bad873c2b38a4bc`).
Twelve capital-city, four opposite-word, four weekday, four young-animal clozes.
Each has a prescribed gold token and a different prescribed incorrect token.
All labels tokenize to one token. No case was dropped. Some contexts are
linguistically underspecified, so these are developmental exact-match labels,
not a public factuality benchmark. The clean-competent subset is reported
separately; its required minimum of 12 was frozen before inference.

For each case, the visible verification input is always exactly the same masked
sequence. Build caches from the gold-filled or wrong-filled sequence. Compare:

- native fresh masking and an independently instrumented plain forward;
- full seed-column override without diagonal correction;
- override with diagonal correction at every layer;
- corrected override at the first layer only (later feedback remains possible);
- corrected override at the final layer only (no later layer can relay it back).

The implementation recomputes the corrected seed attention row directly; it does
not rely on the post-hoc expression. Non-seed queries still use the overwritten
memory. Forward hooks/methods are restored after each case. No token sampling,
instruction tuning, or iterative generation is involved.

## Results

Native and instrumented plain logits match exactly. Last-layer-only corrected
override differs from native by at most **2.003e-5** in logits, below the frozen
2e-4 numerical tolerance, and retains identical answers. This is the important
negative control for an implementation bug masquerading as multilayer leakage.

| Verification condition | Gold-cache correct /24 | Wrong-cache correct /24 | Wrong-cache correct on clean-competent /14 |
| --- | ---: | ---: | ---: |
| Fresh masking (no candidate cache) | 14 | 14 | 14 |
| Override without correction | 12 | 13 | 11 |
| Corrected override, all layers | 16 | 16 | 13 |
| Corrected override, first layer only | 15 | 15 | 14 |
| Corrected override, final layer only | 14 | 14 | 14 |

Changing the cached candidate changes the all-layer corrected top answer on
**2/24 cases**, both outside the clean-competent subset. On 21/24 cases, the
wrong-candidate cache shifts log-odds toward the incorrect token relative to the
gold-candidate cache; mean shift **1.227 nats**, median **.951 nats**. This last
log-odds calculation is a post-hoc descriptive analysis of saved logits, not a
registered success criterion. It shows influence on relative scores, not a
calibrated probability of error.

The two changed cases are the calf and foal clozes. Neither becomes the specified
incorrect token under corrected override; both remain incorrect under our
exact-match labels. In the clean-competent subset, gold and wrong corrected
caches both lose the same one correct answer. Therefore we must not portray
those counts as fourteen clean successes spoiled by wrong cached answers.

The salient practical result is mixed: corrected override scores **better** than
fresh masking overall on this small set. A separate clean pass restores exact
independence but does not improve aggregate accuracy here. No practical fix has
earned a green light, and the source paper's efficiency/quality claims are not
refuted by these measurements.

## Next decision

Keep the mathematical independence finding, but hold the proposed performance
story. Before investing in a full diffusion benchmark, identify a realistic
verification setting with competent controls and naturally occurring mistaken
drafts. Measure whether candidate influence actually changes Keep/Replace
decisions, and compare accuracy at equal verification cost. Do not select only
cases where the new check wins or confuse logit sensitivity with useful repair.
This is planned research, not a live queue or a request to keep a GPU powered on.

## Preservation and tests

Root `artifacts/cache_verification_bert_v1`; full vocabulary logits, per-case
token IDs/outputs, frozen config, model-file hashes, implementation hash and
runtime are preserved. Manifest SHA-256:
`0a4529882eef2e0f9bfbcd014cf2f431c12d286d9fce1753c56e5b399f7deaa9`.

Verifier checks bytes (including the local model), numerical controls and metric
recomputation. It **does not replay every model forward**. Combined operator/BERT
tests: **17 pass**, including native equality, method restoration, last-layer
negative control, first-layer feedback, and the pinned upstream arithmetic.

```powershell
$env:PYTHONPATH='src;.'
python -m pytest tests/test_cache_verification_audit.py tests/test_cache_verification_bert.py -q
python -m interaction_sprint.cache_verification_bert verify artifacts/cache_verification_bert_v1
```

All previous evidence is unchanged. No active CPU/GPU process remains. The
overall goal is still unmet; no paper expansion or acceptance claim is justified.
