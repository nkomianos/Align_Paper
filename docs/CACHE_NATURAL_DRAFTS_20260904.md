# Natural-draft cached verification — developmental protocol

**Completed, but the intended natural-lexical-draft interpretation is invalid.**
All 128 truncated-context drafts were punctuation. The frozen numerical rule
passes, but that cannot validate a premise the rule omitted. A separate format
qualification is complete; the repaired full comparison has not run.

## Question

Does candidate-bearing cache reuse prevent a pretrained masked LM from correcting
its own guess after additional context becomes available? The previous cloze
audit found candidate-dependent scores but no aggregate advantage for a clean
verification pass. This test uses model-generated guesses, not hand-chosen wrong
tokens, and records both recovered mistakes and newly introduced errors.

This remains an independent BERT information-flow test. It is **not** a current
diffusion-decoder benchmark or an evaluation of COVER's full seed scheduler.

## Frozen data and selection, before model inference

[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
`wikitext-2-raw-v1/validation-00000-of-00001.parquet`, repository revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Downloaded only that public validation
file, 657,209 bytes; SHA-256
`204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`
matches the Hub's LFS object hash. Dataset card lists CC-BY-SA-3.0 and GFDL;
retain original article names, source identifiers and license metadata with
derived cases. No private or locked research TEST data was opened.

The frozen configuration is `configs/cache_natural_drafts_v1.json`.
Sentence segmentation uses punctuation/whitespace, not an LM. Eligible sequences
have 20–96 BERT tokens. Choose the alphabetic single-WordPiece word nearest the
midpoint, at least eight context tokens on each side. Require at least three
letters and exclude targets followed by continuation subtokens. Deduplicate
sentences, then sort by a seeded content hash, capped at four cases per article.
No model score or correctness label is used to choose cases.

Prepared **128 cases from 52 articles**. Some targets are ordinary function
words; no semantic-content filter is claimed. Sentence splitting can include
abbreviation artifacts. This is exact corpus-token reconstruction, not a claim
that every nonmatching completion is semantically wrong. BERT may have seen
these Wikipedia texts during pretraining; no unseen-fact generalization claim.

Prepared root `artifacts/cache_natural_drafts_prepared_v1`, manifest SHA-256
`8d47777db8fe50b70feb97e09e12986e9ffd2475afb7cf9224214e0ec88e9656`.
That manifest was created before inference and binds the configuration, selected
cases, source implementation and original parquet bytes.

## Model and interventions

Reuse the local public BERT checkpoint and instrumented attention implementation
from [the previous audit](CACHE_VERIFICATION_BERT_20260904.md), unchanged:
`google-bert/bert-base-uncased` at
`86b5e0934494bd15c9632b12f734a8a67f723594`, CPU float32/eager/eval, two threads.

For each selected position:

1. Hide the target and truncate its right context. Let the model's full-vocabulary
   argmax be its draft. This is not an autoregressive model or sampled rollout.
2. Reveal the original right context while leaving the target masked.
3. Compare eight output conditions below. The source gold token is used only
   for evaluation, never to construct candidate-bearing caches.

| Condition | Cache provided during full-context verification |
| --- | --- |
| Draft | None; prediction from truncated context, before disclosure |
| Fresh | None; native full-context masked forward |
| Plain | None; instrumented full-context masked forward |
| Stale uncorrected | Prefix plus model's draft; no diagonal correction |
| Stale diagonal | Same prefix/draft cache; diagonal correction at every layer |
| Refreshed diagonal | Full disclosed context plus draft; same correction |
| Neutral diagonal | Prefix with target still masked; same correction |
| Last diagonal | Prefix/draft cache injected only at final layer, corrected |

Stale versus refreshed distinguishes changed surroundings from target feedback.
Stale versus neutral distinguishes a candidate-bearing cache from a cache that
never saw the guess. Last-layer-only injection is a negative control: once its
direct seed contribution is removed, there is no downstream layer through which
the candidate can return. All verification inputs are paired and identical.

Each case requires eight scored forwards and three cache-building forwards:
**1,408 CPU forwards planned; no parameter updates.** The experiment is finite,
not a background expansion queue.

## Interpretation fixed before results

Native/plain and native/last-only max-logit differences must be below 2e-4;
native/plain argmax must agree. Otherwise this run fails instrumentation checks.

Primary comparison: fresh minus stale-diagonal total correct. Count fresh-only
and cache-only successes explicitly. Also report all initial errors, those
recoverable by fresh masking, remaining old guesses on that subset, and loss of
initially correct predictions. Do not condition the headline comparison only on
cases where the proposed clean baseline succeeds.

A practical follow-up needs at least 20 fresh-recoverable drafts and at least
five net extra correct cases for fresh versus stale-diagonal verification.
These are developmental triage rules, **not significance thresholds or a paper
acceptance criterion**. Article-cluster bootstrap intervals preserve paired
cases and within-article dependence. No thresholds are fitted on these outputs.

Even a passing comparison would not establish speed-quality improvement: an
actual diffusion decoder combines drafting and verification, which this isolated
test does not time or reproduce. No GPU experiment launches automatically.

## Reproduction

```powershell
$env:PYTHONPATH='src;.'
python -m pytest tests/test_cache_verification_audit.py tests/test_cache_verification_bert.py tests/test_cache_natural_drafts.py -q
python -m interaction_sprint.cache_natural_drafts verify artifacts/cache_natural_drafts_v1
```

The verifier checks source, backend, prepared data, model and output bytes,
per-case argmax records, controls and metric recomputation. It does not replay
all model forwards. New runs require an absent root; no artifact is overwritten.

## Completed numerical result and PI interpretation

All **1,408 CPU forwards** completed, 177.59 seconds for the inference and summary
loop (excluding initial loading and output verification). Native/plain equality
is exact; last-layer-only max-logit error is 2.265e-5, below the frozen tolerance.

| Condition | Correct corpus token /128 |
| --- | ---: |
| Initial draft | 0 |
| Fresh full-context masking | 73 |
| Plain instrumented | 73 |
| Stale uncorrected | 61 |
| Stale diagonal-corrected | 62 |
| Refreshed diagonal-corrected | 59 |
| Neutral diagonal-corrected | 69 |
| Final-layer-only corrected | 73 |

Fresh is correct on 12 cases where stale-diagonal is wrong, with one case in the
opposite direction: net **11/128 = 8.594 percentage points**. The descriptive
article bootstrap interval is **[3.85, 13.99] points**. The frozen runner returns
`DEVELOPMENTAL_CLEAN_GAIN_INVESTIGATE_NOT_PAPER_GO`; preserve that output.

However, inspecting the actual drafts exposes the apparatus flaw: **108 periods,
18 semicolons, one question mark and one exclamation mark**. The truncated
`... [MASK] [SEP]` layout turns the target into a sentence-ending position for
BERT. These are naturally produced outputs of this input, but not a credible
sample of mistaken lexical guesses. The absence of any initially correct draft
also means this run cannot measure damage to initially correct predictions.

Furthermore, stale diagonal correction retains the exact old punctuation token
on **zero** of the 73 fresh-recoverable cases. The errors do not establish the
specific story that an old wrong answer is repeatedly confirmed. The observed
accuracy loss is real for this apparatus, but its practical interpretation is
limited. **PI decision: do not expand on the apparent gain.** No threshold was
retroactively edited and no evidence was discarded.

Complete root `artifacts/cache_natural_drafts_v1`; manifest SHA-256
`7da8b8c731c4592f713ee3e51d5d8074b96c428ed9b8112dcfa3f2c1f388582f`.
Model/source/data/output integrity and metric recomputation pass. Twenty combined
operator, BERT and natural-draft tests pass. This does not make the omitted
lexical-draft prerequisite true.

## Separate post-hoc format qualification

`scripts/qualify_cache_draft_context.py` runs native BERT only: no cache correction
and no competing method scores. On the first 24 already exposed DEV cases,
compare zero versus three visible right-context tokens before the terminal SEP.
The source token remains masked; retained context never includes the target.

| Visible right context | Lexical guesses /24 | Correct /24 |
| --- | ---: | ---: |
| 0 tokens | 0 | 0 |
| 3 tokens | 23 | 13 |

These **48 additional CPU forwards** qualify a candidate format; they are not
held-out confirmation or evidence of a corrective method. The next comparison
must freeze the short-right-context design, exclude these 24 DEV cases from
its primary evaluation, and test both repairs and regressions. Keep the full
comparison's outcome open; do not assume the 8.594-point difference will persist.

DEV root `artifacts/cache_draft_context_dev_v1`, manifest SHA-256
`090d3818c70e9b2304afab289f11bebedd02d17308e4877436b2cf82d6e4d43f`.
Source/model/prepared-data/output hashes, 48 unique case-condition records and
metric recomputation checked. No full model-forward replay claimed.

A possible later small-model route is [OPDLM-0.6B](https://huggingface.co/divelab/OPDLM-0.6B),
whose authors describe a Qwen3-derived, four-token block diffusion model. Only
its model card was inspected: no weights downloaded, native runner validated,
or experiment queued. Its blockwise context differs from full-attention BERT
and must be respected rather than silently imposing our current apparatus.

All runs have exited. No paid GPU or automatic expansion is active.
