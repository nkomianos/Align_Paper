# Hindsight: what a fair full-sequence follow-up would require

Research design only, 4 September 2026. No implementation or GPU launch is
authorized by this document. This qualifies the earlier allocation audit: the
current numeric-choice apparatus should remain parked, but it is not a fair
empirical rejection of the original report's broad interaction-learning idea.

## What actually ran

`parameter_probe.py` defines Qwen/Qwen3-0.6B at revision
`c1899de289a04d12100db370d81485cdf75e47ca`; the factorial and competence studies
import that same model. They use restricted A/B distributions, not native
full-sequence response generation. One study takes one normalized gradient step;
another takes no updates; the competence pilot takes real adapter updates but
uses an exact known-channel expectation and frozen binary teachers. The older
Qwen3-4B study performs 512 forward passes, not parameter training. None is a
full-sequence reproduction of released online SDPO on a qualified model.

The cached Qwen3-4B snapshot
`1cfa9a7208912126459214e8b04321603b3df60c` is therefore a sensible continuity model
for a genuine method-feasibility pilot. It is not a current frontier-family
confirmation and cannot establish a general alignment claim alone.

## Fidelity requirements from primary source

The [released online updater](https://github.com/lasgroup/user_interactions/blob/3b17d2a67bd2565b9fbda495fd16a485406aa954/online_sdpo_updater.py)
supports native generated responses and LoRA updates. Its simple-signal loss
uses stopped per-token hindsight/base log-probability differences, completion
length normalization and a refreshed teacher before each update. The original
[SDPO interaction paper](https://arxiv.org/html/2603.12273v1) studies both offline
interactions and continual personalization. The proposed follow-up should
exercise this sequence-level mechanism, not replace it with a binary teacher.

Use its pinned prompt block and loss, explicitly configuring clipping,
temperature, EOS, ignore-first-token setting, teacher refresh and optimizer.
No vLLM/multiple-GPU requirement is necessary for this small pilot. Native
generated IDs must be retained; the upstream string re-tokenization and EOS
addition should be audited against those IDs rather than silently assuming
equivalence. Disclose any engineering departure needed for exact ID handling.
No top-k loss approximation or alternate KL direction should be relabeled the
released simple-signal objective.

## One bounded, interpretable pilot

Ask whether full-response SDPO can learn stable user-specific formatting from
truthful corrections, and whether action-dependent polite approval changes
that learning. This isolates **expression endogeneity**, not persistent
preference transition. Do not call expression copying persuasion.

Eight opaque user IDs have fixed preferences across four equally represented
response formats: compact JSON, a two-item bullet list, a two-column table,
and a short labeled plain-text response. Each response reports two facts from
a supplied micro-document. The preferences are absent from the ordinary input;
the ID is present, allowing personalization across episodes. The model cannot
be expected to know an unseen ID's preference before feedback.

Use deterministic syntax and exact fact-preservation checks, frozen before
generation. JSON is validated as JSON; tables and bullets have declared grammar.
Score the entire response, counting malformed or truncated responses as such.
Do not retrofit permissive answer extraction after seeing model outputs.
All formats can satisfy content correctness. This avoids conflating the preferred
style with an objectively correct factual answer.

Calibration, adaptation and heldout evaluation use different facts, documents
and request wordings. Evaluate heldout tasks for the same user IDs: this is
within-user personalization, not unseen-user generalization. User-level paired
statistics must disclose only eight user units; hundreds of tokens are not
independent human participants.

### Qualification before training

On 32 separate calibration tasks, explicitly state the preference and require
at least 90% joint format/content validity. Separately start from each task's
ordinary completion and condition on a truthful corrective future message;
require at least 90% content validity and substantial format recovery among
initial mismatches. A format with no mismatches has a ceiling, not a failure.
Report all format-specific counts; do not choose only easy profiles afterward.
Do not inspect final heldout outcomes to select the model or checkpoint.

The model must also show nontrivial ordinary preference errors before adaptation;
otherwise there is no personalization headroom. This is expected because user
IDs are opaque, but it must be measured, not assumed. If calibrated explicit
preferences or the hindsight-conditioned teacher fail, stop before training
and label the apparatus unqualified.

### Positive control before the endogenous-feedback comparison

Run 64 fixed online updates (eight episodes per user), fresh model-generated
responses of at most 64 tokens, temperature and optimizer frozen in advance.
Truthful feedback says whether the output format matches the user's fixed
preference and provides the desired format when it does not. The content task
remains unchanged. Evaluate fixed final adapters against the no-adaptation
checkpoint on novel tasks for those users.

If truthful-feedback SDPO does not improve preference adherence while preserving
fact accuracy, do not interpret a subsequent harmful/noisy-feedback run. The
method has not demonstrated useful adaptation in this apparatus. An anchor-SFT
positive reference may diagnose learnability, but is extra supervision, not an
SDPO reproduction or our proposed new corrective algorithm.

### Endogeneity experiment, only after the positive control succeeds

From the identical base checkpoint, run the same fixed episode budget under
an expression-only channel. On a predeclared fraction (for example one half)
of format mismatches, a structured user emits polite approval instead of the
truthful correction. On all other episodes it responds truthfully. The hidden
preference never changes. Log the random channel draws and feedback provenance.
This channel changes what is observed, not the scoring label.

Compare no adaptation, truthful-feedback SDPO and this expression-dependent
SDPO; report initial/final preferred-format adherence, fact accuracy, feedback
approval frequency and calibration. Approval can increase mechanically under
the channel: that increase alone is not learning or evidence of the paper's
self-reinforcing failure. Require a change in the trained policy relative to
both its own initial checkpoint and the truthful-feedback control. A smaller
benefit is not the same claim as degradation below no adaptation.

This experiment can produce no divergence or even improvement. Those outcomes
remain valid. Do not tune the approval rate, learning rate, profiles or number
of steps to manufacture a harmful curve.

## What this can and cannot decide

It could close the real engineering gap: qualified 4B model, native full
responses, actual released-style learning, beneficial-correction control, and
a precisely defined action-dependent expression intervention. It would not
identify preference transitions in real users or establish that current
interaction-learning systems generally manipulate users.

The broad issue is already addressed by
[Privileged Likelihood Is Not Automatically Value](https://arxiv.org/html/2608.09263v1).
A successful pilot still needs a useful, distinct correction and strong
information-budget-matched baselines. Adding privileged anchors and beating
an unanchored learner is insufficient; anchor-only supervised learning is a
required comparator. No new corrective method has been established here.

## Remaining-window feasibility

Conditional **yes for a developmental feasibility test**, not a paper result:
cached 4B weights, short prompts, 64-token responses and 64 updates per arm are
plausibly within a few GH200 hours. Reserve 2-4 hours as a planning envelope
including calibration, positive control, second arm and evidence transfer;
this is not a measured runtime estimate. Time a native response and complete
teacher/student backward update before committing the remainder. Prefer early
qualification stop over spending the whole window on an invalid apparatus.

Priority remains conditional on the independent UNDO run and remaining time.
Do not displace an already running scientifically qualified experiment merely
to fill the queue. No source code was produced by this design-only audit.
