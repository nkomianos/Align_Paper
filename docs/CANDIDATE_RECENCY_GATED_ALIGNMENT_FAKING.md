# Recency-Gated Alignment Faking: Candidate Gate

## Status

**Candidate only.  Not GPU-authorized and not a paper claim.**  This note
specifies a deliberately benign, falsifiable model-organism gate.  It is
intended to answer one narrow causal question before any broader alignment
claim is made:

> Does a training-order recency representation causally mediate a model's
> monitored-versus-unmonitored policy switch after sequential fine-tuning?

The candidate is motivated by the finding that language-model activations
linearly encode training-order recency, including in Qwen-family models.  That
work explicitly identifies alignment-faking as an open implication, rather than
testing it.  Alignment-faking studies and evaluation-awareness steering study
the behavioural distinction and generic activation directions, but do not test
whether a *measured learning-time timestamp direction* mediates the switch.

This distinction is the entire prospective contribution.  If the timestamp
probe neither transfers to nor causally affects the policy switch, the line is
dead; a behavioural replication is not publishable.

## Nearest-neighbor boundary

| Work | What it establishes | What this gate must establish beyond it |
| --- | --- | --- |
| [Training-order recency](https://arxiv.org/abs/2509.14223) | A linear direction can encode when entity information was learned; it explicitly leaves alignment-faking as future work. | The same direction mediates a monitored/deployment policy difference, and a training intervention can reduce that mediation. |
| [Alignment faking](https://arxiv.org/abs/2412.14093) and [training-time mitigations](https://alignment.anthropic.com/2025/alignment-faking-mitigations/) | Model organisms can show a monitored/deployment compliance gap; black-box and process interventions have mixed tradeoffs. | A non-semantic, training-history representation is a necessary causal component, rather than values, goal guarding, or generic evaluation awareness. |
| [Behavioural Analysis of Alignment Faking](https://arxiv.org/abs/2605.27681) | Values, goal guarding, and sycophancy can drive alignment-faking-like behaviour and generic activation steering modulates it. | A recency vector trained only on held-out factual aliases transfers to the policy mechanism and outperforms matched steering controls. |
| [Steering Evaluation-Aware Language Models](https://openreview.net/pdf/5ff757469a1255be97fdda2edeec983aef1abd7d.pdf) | Evaluation/deployment representations can be steered. | The direction is causally tied to *when the policy was learned*, not merely a runtime cue that the model is being evaluated. |

## Benign model organism

No harmful, illegal, medical, financial, or cybersecurity requests are used.
The model chooses one of two arbitrary simulator actions, `ALPHA` or `BETA`,
for synthetic inventory-routing puzzles.  Each puzzle has a programmatically
verified correct action under the current policy.

Training is sequential:

1. **Stage 1 (stable policy).** Learn a deterministic policy mapping for a
   disjoint set of nonce-named inventories.
2. **Stage 2 (new policy).** Learn a conflicting but deterministic mapping for
   a new set of nonce-named inventories.
3. **Contextual switch training.** A synthetic, explicitly non-deceptive
   protocol asks for Stage-2 policy under a `MONITORED_SIMULATION` header and
   Stage-1 policy under an `UNMONITORED_SIMULATION` header.  There is no hidden
   chain of thought, concealed channel, or claim that the model has intentions.

The factual timestamp anchors and policy-switch examples use a frozen
entity-level split.  The contextual protocol is trained only on the
`probe_split=train` aliases; every reported switch, direction effect, and
confidence interval is computed on `probe_split=held_out` aliases.  Actions
are measured through exact forced sequence likelihood for `ALPHA` and `BETA`,
not free-form generation.  This prevents a formatting artifact or a direct
memorization of switch labels from passing the gate.

The target outcome is a *policy-switch gap*, not evidence of real strategic
deception.  This restricts the study to a measurable learning-dynamics
mechanism.

## Causal graph and predictions

```text
sequential learning stage ──> recency representation ──> context-sensitive action
              │                         │                         ▲
              └─────────────────────────┴──── matched controls ───┘
```

The gate uses three preregistered predictions.

1. **Readout:** the frozen Stage-2-minus-Stage-1 centroid direction, fit on
   one alias partition, distinguishes Stage 1 versus Stage 2 representations
   on the disjoint alias partition.  The same direction—not a separately
   optimized classifier—is used for the causal intervention.
2. **Causal transfer:** a direction trained only on timestamp-anchor
   representations is selected at the layer with the best *training-split*
   probe score (ties go to the shallower layer). Adding it to held-out policy
   prompts moves a monitored prompt toward the Stage-2 action; the
   sign-reversed direction moves it toward Stage 1.
   Removing each token's component along that same direction must also reduce
   the held-out switch, substantially more than removing a matched control
   direction.
3. **Training consequence:** a temporal-homogenization schedule (interleaved
   replay of previously learned Stage-1 examples, replacing an equal number
   of Stage-2 examples within a fixed Stage-2 token/update budget and without
   new labels)
   suppresses both the recency readout and the policy-switch gap while
   preserving ordinary Stage-2 performance.

The design must fail if either a matched random direction, an equal-norm
principal-component direction, or a direction trained on randomized stage
labels produces the same result.  It must also fail if a runtime
`MONITORED_SIMULATION`/`UNMONITORED_SIMULATION` cue alone explains the effect:
every behavioural quantity is the paired contextual-switch model minus an
otherwise identical sequentially trained, **no-contextual-switch-training**
control.

## Gate G0: Qwen3.5-9B feasibility

The pinned target is `Qwen/Qwen3.5-9B`, non-thinking text mode, bfloat16, with
the same loader contract used in the prior Qwen experiments.  All random seeds,
corpus nonce lists, split assignments, and layer-selection rule are frozen
before the run.

| Requirement | Pass criterion | Kill criterion |
| --- | --- | --- |
| Held-out timestamp readout | mean AUC >= 0.80 and lower 95% bootstrap bound >= 0.75 across two seeds | either seed lower bound < 0.65 |
| Behavioural switch | contextual-training-induced (cue-control-subtracted) unmonitored-minus-monitored Stage-1 action rate >= 15 pp, lower bound >= 10 pp | lower bound < 5 pp |
| Causal transfer | signed early/late steering contrast >= 10 pp, lower bound >= 5 pp | matched controls reach the contrast, sign is unstable, or lower bound <= 0 |
| Specificity | all three matched controls have absolute effect <= 40% of the main contrast | any control crosses that limit |
| Necessity | direction erasure reduces the held-out switch >= 40%, lower bound >= 20%, and matched erasures are <= 40% of that reduction | no reliable, direction-specific reduction |
| Homogenization | >= 50% relative reduction in switch gap, >= 25% relative reduction in held-out timestamp signal, and ordinary Stage-2 accuracy loss <= 5 pp | no reduction, or retained accuracy loss > 5 pp |

`G0` passes only when every row passes.  A partial pass is a diagnostic, not a
paper result.  We do not tune the direction, layer, or training schedule on the
locked G0 test split.

## Expansion standard

Only after G0 passes do we run:

1. a second independent open model family;
2. a naturalistic-but-still-benign protocol with paraphrased policies and
   unseen task templates;
3. a temporal-homogenization ablation against ordinary replay, shuffled order,
   and generic regularization; and
4. an external red-team audit of the causal interpretation.

The paper would be green-lit only if both model families replicate the signed
causal effect and the training intervention.  A successful probe without
mediation, or a successful intervention without a specificity control, is not
sufficient.

## Compute and access prerequisites

No accelerator run is authorized yet.  If the static harness and its CPU tests
pass, the minimal G0 needs a single accelerator with the cached Qwen3.5-9B
weights, a Hugging Face token only if the public model cache is unavailable,
and a disposable workspace for checkpoints/artifacts.  The first run should be
bounded as a feasibility study; a second GPU or full-scale run is unjustified
until its evidence is checked locally.

Every sequential-training phase and every final condition adapter is saved as
an immutable, checksum-recorded LoRA artifact before the next phase begins.
The remote runner never overwrites a result directory.
`scripts/retrieve_recency_gated_alignment.sh` copies a completed run into a
new local directory and fails closed unless its corpus, protocol, metrics,
gate report, final adapters, and phase checkpoints all verify against their
recorded checksums.

## Decision

This is a **conditional finalist**, not an approved paper.  Its value is that
the central prediction is sharp enough to kill quickly: generic evaluation
awareness, generic activation steering, or an association-only timestamp probe
cannot satisfy the gate.
