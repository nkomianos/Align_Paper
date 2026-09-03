# Idea review — 2 September 2026

## PI recommendation

Prioritize **Efference-Pair's synthetic apparatus pilot**, not a large training
run. Keep **visual hindsight leakage** as the already implemented independent
backup. Investigate **latent-interface compatibility after model updates** as a
new reserve, but do not rent a GPU for it yet. None is a confident acceptance
prediction; each has explicit failure conditions and substantial novelty risk.

## What the Mostik article does and does not establish

The [supplied Techbuzz story](https://www.techbuzz.ai/articles/mostik-teaches-ai-models-to-skip-language-talk-directly)
is a scouting lead, not experimental evidence. [Mostik's own site](https://mostik.ai/)
describes internal-representation communication but the inspected page provides
no reproducible benchmark, implementation or weight release. Do not infer its
performance or claim it invented communication without text.

This general direction already has strong primary precedents:

| Primary source | What it already covers | Consequence for us |
| --- | --- | --- |
| [Cache-to-Cache, ICLR 2026](https://github.com/thu-nics/C2C) | Learned cross-model KV projection/fusion, with public code and fusers | Generic latent exchange is not new; published fusers offer a reproduction starting point |
| [Interlat, ACL 2026](https://github.com/XiaoDu-flying/Interlat) | Hidden-state communication, compression, heterogeneous agents and perturbation ablations | Neither compression nor “do messages contain useful information?” is enough |
| [Latent Cache Flow, May/June 2026](https://arxiv.org/html/2605.22863v2) | Smaller cache adapters and controlled cross-context communication; explicitly leaves broader agents/topologies and scales to future work | A named future-work item is motivation, not an automatic novelty claim |
| [LCGuard, May 2026](https://arxiv.org/html/2605.22786v1) | Privacy-preserving latent/KV exchange against reconstruction; limitations include fixed topologies and decoder-dependent leakage assessment | Do not propose a generic latent privacy filter as new |
| [Causal audit of latent communication, July 2026](https://arxiv.org/abs/2607.26773) | Separates example-specific message content from other effects using causal controls | Generic message-swapping audit is occupied |
| [StateBridge, August 2026](https://arxiv.org/abs/2608.13317) | Training-free hidden-state alignment | A simple Procrustes bridge is already a method, not our invention |
| [Compatible representation learning](https://arxiv.org/abs/2509.16664) | Relaxed-orthogonality alignment across model updates | “Representations drift; align them” is also prior work |
| [Verifiable agent semantics](https://arxiv.org/abs/2602.16424) | Shared-stimulus certification, recertification and renegotiation | Generic semantic checksums/certification would overlap directly |

Sources were inspected on this date. This is a bounded search, not proof of
absence of competing work. Search results were followed to primary papers/code;
press descriptions and third-party summaries were not used as scientific evidence.

## 1. Efference-Pair — current work

**Question:** does a frozen VLM fail to subtract camera-induced image motion,
and can showing estimated global motion and camera-compensated residual motion
separately repair that failure at a matched visual budget?

**Nature inspiration:** separating self-induced sensory change from external
change. This is only an analogy: a video observer receives no actual motor
command, so our estimated global motion is not a biological efference copy.

**Why it might matter:** video assistants, robotics perception, and evaluation
of generated motion can confuse a moving viewpoint with a moving object. A
successful frozen interface could be cheaper than post-training each VLM.

**Collision boundary:** [ACaM](https://1yuwen.github.io/ACaM-Project-Page/) studies
camera confusions and training; [4DP-QA](https://research.nvidia.com/labs/lpr/4dpqa/)
already uses a fixed camera reference and motion tracking. Flow visualization is
also established by [Time Blindness](https://timeblindness.github.io/).
Our remaining hypothesis is the *specific global/residual selectivity and
controlled intervention*, not the novelty of optical flow or stabilization.

**Next experiment:** [EP0](EFFERENCE_PAIR_EP0_RUNBOOK.md), with an RGB-only
estimator and separate simulator-oracle diagnostic. A first 24-forward smoke
checks that the VLM can consume the interface and measures speed. The separately
launched 450-forward synthetic pilot measures accuracy and control contrasts.
A pass only warrants designing/sealing the real-video gate. It does not establish
real-world generalization, estimator robustness, or an ICLR paper.

**Scientific traps:** planar motion does not identify general 3-D camera pose;
a stationary background is an assumption; constant-velocity frame shuffling is
not an informative temporal intervention; obvious arrows can reduce the task to
reading a classical estimator. The formal study therefore still needs real
videos, motion-boundary and tracking baselines, and a second model family.

## 2. Silent latent-interface regression after a model update — new reserve

**Question:** can an update that preserves a sender's ordinary text-task
performance break an already trained communication interface, and can a small
unlabelled compatibility calibration recover communication without retraining
the bridge or falling back to text on every item?

This is a proposed extension motivated by fixed-model communication studies,
**not** a result reported by Mostik, and **not** an established novelty claim.
Its potentially interesting object is *behavior-preserving but communication-
breaking updates*, not arbitrary corruption or the trivial fact that changing
latent coordinates can break a decoder.

### Cheap necessary test

1. Reproduce a published bridge on its original small-model pair. Establish that
   the receiver actually gains task information over no message, random message,
   and another example's message. If that fails, stop: an unformed communication
   channel cannot test update robustness.
2. Freeze that bridge and receiver. Apply ordinary, independently seeded,
   low-rank sender updates using disjoint TRAIN data. Screen their text behavior
   on DEV only; require utility within 2 pp of the original. Do not deliberately
   rotate hidden states and call the resulting breakage a discovery.
3. On a held-out communication set, compare old/new sender through the same
   bridge, text transport, no message, and recomputed alignment. Include matched
   sender-only degradation in a difference-in-differences analysis.
4. Prototype a small calibration using unlabelled paired prompts. Compare it
   against re-fitted StateBridge/orthogonal/affine alignment, an equal-data
   bridge refresh, and conservative version-invalidated text fallback. Charge
   all calibration, encoding, transport and fallback time.

**Proposed G0 bar (not yet frozen):** in at least two independently seeded
behavior-preserving updates, a >=10 pp communication loss beyond text degradation;
calibration recovers >=75% of that loss on held-out tasks while retaining a
measured latency advantage. Bootstrap at example and update level, not token
level. Passing one tiny model pair would authorize only another pair/real task.

**Kill/park conditions:** failure entirely explained by ordinary capability
loss, mismatch detectable by version ID alone with cheap adequate fallback,
simple existing re-alignment works equally well, no formation of a useful
baseline channel, or a directly overlapping study. A performance regression
benchmark without a principled remedy is likely too weak for ICLR.

**Readiness:** a label-separated, nonce two-hop communication fixture and its
exact oracle are included under `src/latent_contract/`. This is a CPU data/control
test only. Published bridge integration, model/update pins, matched text budget,
and end-to-end training/inference are **not implemented yet**. Do not call it GPU
ready, and do not use an older model merely to claim a 2026 frontier result:
small published pairs are for replication, newer independent pairs for confirmation.

## 3. Visual hindsight leakage — independent prepared backup

**Question:** does a counterfactual future change a VLM's answer about an already
fixed past decision, even though the two videos have byte-identical prefixes?
The [existing protocol](CANDIDATE_VISUAL_HINDSIGHT_LEAKAGE.md) uses matched suffixes,
prefix-only controls and endpoint visibility checks. This is not a new idea
invented during today's search; code already exists and its CPU tests are rerun
in this update. It has no GPU result yet.

**Why not jump immediately to training?** A small paired effect is easier to
interpret than a failed training pipeline. But ambiguous questions or an unseen
endpoint invalidate the measurement. Even a positive result must beat trivial
prefix clipping as a useful contribution, not just illustrate a benchmark trap.

## Allocation decision

First rent one GPU for EP0 smoke, then use actual throughput to decide the
450-forward pilot. No multi-GPU instance or new token is required for the public
Qwen model. Do not start all ideas in parallel simply because compute is available.
Reserve latent-interface work for a verified baseline bridge; avoid paying for a
long run whose scientific prerequisites have not yet been established.
