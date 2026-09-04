# Cached verification with short right context — fresh-sentence replication

## Frozen question and scope

Does clean verification beat corrected seed-KV caching after a pretrained masked
LM makes a meaningful lexical guess from incomplete context? The preceding run
failed that premise because its target was placed immediately before SEP and
the model always guessed punctuation. Preserve that run unchanged.

This is a new developmental comparison, **not** a fresh hypothesis tested without
prior exploration. It uses BERT, not a modern diffusion decoder, and cannot by
itself establish a speed-quality improvement or an ICLR-ready contribution.

## Before inference

- Use the same pinned WikiText-2 raw validation file and BERT checkpoint as the
  [previous public-text experiment](CACHE_NATURAL_DRAFTS_20260904.md).
- Exclude **all 128 previously tested sentences**, including the 24 format-DEV
  cases. Select new cases by the prior outcome-free eligibility rules, a new
  fixed seed, and at most four sentences per article.
- Freeze **128 cases from 50 articles**, with zero sentence overlap. Articles
  may overlap the earlier dataset; this is not external-domain replication.
- Retain exactly three right-context tokens in the initial guess input, then
  reveal the remaining context. The gold target stays masked in every scoring
  input and is never used to construct the candidate-bearing cache.

Configuration: `configs/cache_short_context_v1.json`.
Prepared root: `artifacts/cache_short_context_prepared_v1`.
Prepared manifest SHA-256:
`243931e810cb2e82d4ac7c2f511226dd51347a506fcf4c189337da1eee157594`.
It binds the new config/cases, prior exclusion source, original dataset bytes,
runner, unchanged selection/metrics code, and unchanged BERT instrument.

## Design

Same eight scored conditions as before: initial draft, native fresh masking,
instrumented plain masking, stale cache without correction, stale cache with
diagonal correction, refreshed candidate cache with correction, neutral masked
cache with correction, and final-layer-only corrected cache. Three cache-building
forwards per case give **1,408 planned CPU forwards, zero parameter updates**.

Primary comparison is fresh minus stale-diagonal total correct. Report both
directions of paired disagreement, repair of initial errors, and damage to
initially correct guesses. Fresh-recoverable cases are a diagnostic subset,
not the sole basis for the headline accuracy comparison.

Prerequisites fixed before inference:

- At least 80% lexical drafts (ASCII alphabetic full token strings).
- At least 16 initially correct and 16 initially incorrect cases.
- At least 20 initial errors recoverable by fresh masking for the practical
  comparison to be informative.
- Native/plain and native/final-layer-only logits agree within 2e-4; native and
  instrumented plain argmax predictions are identical.

Five net additional correct cases for fresh verification is the unchanged
developmental triage criterion. The script retains the numerical decision but
overrides it to invalid if the newly explicit draft prerequisites fail. This is
not a significance threshold. Article-cluster bootstrap intervals are descriptive.

No filtering based on outcomes, no vocabulary restriction to force lexical
guesses, and no relabeling nonmatching words as hallucinations. Exact corpus
reconstruction can penalize valid alternative completions. Wikipedia pretraining
overlap remains possible.

## Reproduction and preservation

```powershell
$env:PYTHONPATH='src;.'
python -m pytest tests/test_cache_short_context.py -q
python -m interaction_sprint.cache_short_context verify artifacts/cache_short_context_v1
```

The verifier checks byte integrity, source/model/data pins, sentence exclusions,
per-case token records, controls and metric recomputation—not all model forwards.
The earlier runner and backend have not been edited, so their frozen evidence
remains verifiable. No existing artifact is overwritten. No automatic GPU
expansion or experiment queue is configured.

## Completed result

All 1,408 CPU forwards completed in 164.33 inference seconds, with no updates.
There are 119/128 lexical guesses, 43 initially correct and 85 initially wrong.
Thus the repaired draft-format prerequisites pass.

| Condition | Correct /128 | Initial errors recovered | Initially correct lost |
| --- | ---: | ---: | ---: |
| Initial draft | 43 | 0 | 0 |
| Fresh masking | 60 | 19 | 2 |
| Uncorrected stale cache | 48 | 13 | 8 |
| Corrected stale cache | 52 | 15 | 6 |
| Corrected refreshed candidate cache | 51 | 15 | 7 |
| Corrected neutral cache | 59 | 19 | 3 |
| Final-layer-only corrected cache | 60 | 19 | 2 |

Fresh-only correctness occurs on ten cases, cache-only on two: net +6.25pp.
Descriptive article-bootstrap 95% interval: +1.48 to +11.38pp. Corrected stale
cache retains the old wrong guess in four of 19 fresh-recoverable cases.
Native/plain logits match exactly; final-layer-only error is 2.19e-5.

Frozen decision: `INSUFFICIENT_RECOVERABLE_DRAFTS_NO_PRACTICAL_GATE_DECISION`
because there are 19 recoverable initial errors, not the required 20. This is
not a scientific discontinuity at 20, nor a reason to erase the observed effect.
Do not tune the sample count to obtain a pass. This is preliminary evidence
of a practically measurable dependency in BERT, not a modern-decoder method
result. A different architecture and actual draft/verification tradeoff are
more useful next steps than chasing this screening cutoff.

Evidence manifest SHA-256:
`26e7ce4f2a22f07c5d8c5b7c8145600b4853b45f1340416218b5017208cb42d4`.
The committed verifier passed byte integrity and metric recomputation (not
forward replay). All 23 combined operator/BERT/natural/short-context tests pass.
All evidence is local; no GPU used and no process remains active for this run.
