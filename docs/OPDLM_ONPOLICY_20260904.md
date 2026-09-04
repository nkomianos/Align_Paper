# OPDLM on-policy cache verification — frozen developmental comparison

## Purpose and limits

Previous work qualifies OPDLM-0.6B locally but unprompted Wikipedia exact-token
reconstruction was too weak to judge utility. This experiment uses actual chat
generation, model-produced context, and task outcomes. It is still a small
developmental study: not an official COVER benchmark, new SOTA, or paper go.

Model/source/runtime follow [the qualified implementation](OPDLM_CACHE_DEV_20260904.md).
24 fixed arithmetic prompts (8 addition, subtraction, multiplication each),
one model, deterministic greedy confidence order, 32-token cap. No training,
no gold fed into generation, no sample filtering, no paid GPU. Configuration
`configs/opdlm_onpolicy_v1.json`; source/config/model hashes frozen in output
`FROZEN.json` before any forward. Max 3,648 forwards including diagnostic probes,
but EOS can end each policy much earlier. No automatic expansion.

## Four closed-loop policies

Each four-token block begins all masked. Native prediction commits the most
confident token (seed); the next native pass commits a second token and saves
the seed's K/V from that pass. These are genuinely model-produced candidates.

1. **Baseline:** ordinary native third/fourth token drafting; never revise seed.
2. **Fresh:** at the third pass, remask the seed. Use its fresh prediction to
   revise it and the same pass to draft the third token. Draft fourth normally.
3. **All-layer corrected cache:** same as fresh, with saved seed K/V injected at
   every layer and the verification row corrected exactly within each layer.
4. **Final-layer corrected cache:** inject only at the final attention layer,
   preserving clean single-seed verification while optionally informing drafting.

All four policies have four scheduled forwards per complete block. This does
not mean equal latency: cache collection/overrides/row corrections add work.
Fresh is a deliberately simple simultaneous drafting/verification baseline, not
the more expensive option of separate full-context drafting plus clean checking.
No claim that these heuristic policies reproduce COVER's verifier/selection rules.

On the **baseline's shared trajectory**, also evaluate fresh, all-layer and
late-only verification at the same post-second-token state. These three extra
forwards per baseline block are *diagnostic overhead*, not decoder deployment
cost. Retain paired seed distributions, probability shifts, changes of argmax,
and whether candidate caching retains a seed that fresh checking rejects.
Rejection is not automatically correction: a model's own answer is not ground truth.

## Outcomes fixed before inference

- Strict answer: complete stripped output must match a signed integer, and equal
  the arithmetic result. Report parse success separately.
- Secondary audit: last standalone numeral matching the result, always labeled
  permissive and never substituted for strict scoring. This parser can miss or
  misread prose/decimals; neither metric is silently repaired after outputs.
- EOS completion and seed revision counts by policy.
- Same-state all-cache versus fresh seed argmax and candidate probability changes.
- Final-layer versus fresh seed logit agreement (<3e-4 numerical prerequisite).
- Final-layer changes of the two still-masked drafting positions.

Report all policies, both gains and regressions, and exact paired task counts.
No pass/fail significance cutoff for this small diagnostic. A candidate method
needs robust task benefit over simple alternatives, acceptable overhead, further
families/tasks and a defensible novelty claim before paper expansion. A pass of
instrumentation or evidence hashes is not scientific success.

Evidence root `artifacts/opdlm_onpolicy_v1`; checksummed immutable raw outputs,
paired logits and runtime metadata. Verifier recomputes metrics, token decoding,
prompt/answer correspondence and counts; not full model forward replay.

## Prior-work check during the frozen run

[Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/html/2606.12232v1)
(10 June 2026), section 2.3 footnote 2, explicitly notes that shadow verification
is exact for one layer but indirect candidate information persists in deeper
models. Section 3.2 actually compares shadow and full leave-one-out checking,
finding similar behavior in its tested setting. Thus neither the general leakage
observation nor an independent-verifier comparison is a novel contribution here.
The same paper finds little extra value from remasking under short-block greedy
decoding and calls for evaluation of other methods including COVER.

This narrows, rather than strengthens, our paper case. COVER-specific cache
semantics differ from shadow tokens; a reproducible important failure and a
useful correction could still matter, but our elementary witness and BERT
experiment alone do not justify a new-paper claim. Completing the frozen small
run remains useful for deciding whether the proposed correction has any task
benefit. Do not change its policy, tasks or metrics in response to this paper.

## Separate analysis prepared while the run is live

`scripts/analyze_opdlm_onpolicy.py ROOT FRESH_REPORT.json` first invokes the
frozen verifier, then reports task-paired wins/losses and descriptive bootstrap
intervals for both scoring rules. Exact discordance p-values are unadjusted,
exploratory, and are not used as an acceptance threshold. A zero-width empirical
bootstrap interval for identical small-sample outcomes does not prove equivalence.

It also separates seed tokens that belong to the last parsed answer numeral
from earlier numerals (including echoed operands), special tokens, and other
text. This is a post-hoc coverage diagnostic, not a new primary metric or a
change to seed selection. It checks whether this particular highest-confidence
selection policy is actually verifying answer digits before drawing conclusions
about correction. The frozen experiment is still unchanged.

An explicitly exploratory threshold audit additionally counts both directions
of decision disagreement at candidate probability 0.8, the threshold used in
the published WINO re-evaluation. This is not a tuned cutoff or an actual
threshold-policy experiment: our deployed variants revise by argmax. It checks
whether unchanged argmax can conceal potentially consequential confidence shifts;
it cannot establish that threshold remasking would improve final answers.

## Completed result and PI decision

All 96 policy/task outputs completed: 1,661 CPU forwards, 1,160.05 seconds
(19.33 minutes), no weight updates. Session 72509 exited successfully. Frozen
verifier passed source/model/evidence hashes and recomputed scores, tokens,
probabilities and counts; it does not replay model inference. Manifest SHA-256:
`52d3742bb799faf995d16ef76b5d7e282ddc382e7eb4d163cc37f4a764fd49ba`.
Separate analysis: `artifacts/opdlm_onpolicy_v1_analysis.json`.

| Policy | Strict correct / 24 | Permissive last-number correct / 24 | EOS / 24 |
| --- | --- | --- | --- |
| Ordinary baseline | 0 | 24 | 24 |
| Fresh checking | 0 | 24 | 23 |
| All-layer corrected cache | 0 | 23 | 24 |
| Final-layer corrected cache | 0 | 23 | 23 |

All strict failures are answer-only format failures: the model emits equations
and/or explanations. Preserve this result, but do not misdescribe it as zero
arithmetic competence. Baseline is at ceiling under the separately declared
permissive metric, leaving no opportunity to demonstrate answer correction.
All-layer cache changes 91-47 from 44 to 41; final-layer cache changes 9-4 from
5 to 55. Neither cache policy wins any task over baseline. These are small-sample
regressions, not population-level proof that caching is harmful.

Of 87 same-state probes, all-layer versus fresh seed argmax differs twice,
both on non-numeric tokens; one retains a candidate fresh checking rejects.
Only 18 seeds overlap final answer numerals, versus 41 earlier numerals,
19 other tokens and 9 special tokens. Final-layer checking agrees with fresh
seed logits to 7.63e-6, while changing three remaining draft argmaxes.
This confirms the implemented single-seed isolation property, not task utility.
At the exploratory 0.8 threshold, disagreement occurs twice in each direction;
no threshold-based policy was run. No speedup claim follows from call counts.

**PI: park this proposed correction; do not expand it on the paid GPU.**
There is no demonstrated benefit, answer-seed coverage is limited, task accuracy
is at ceiling, and indirect leakage itself is already explicit prior work.
This does not refute COVER or every cache-verification improvement. Reopening
would require a distinct contribution and a justified non-ceiling task, not
more samples of this assay. Seven targeted tests pass; raw evidence and all
earlier attempts remain unchanged. The fallback novelty triage records broad
ideas rejected for overlap, not additional runnable experiments.
