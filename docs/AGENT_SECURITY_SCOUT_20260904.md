# Agent-security mechanism scout — 2026-09-04

## Decision

**NO-GO for adding a new GPU experiment from this bounded scout.** This is a novelty/readiness decision, not a negative experimental result. No GPU run, attack optimization, or external action was performed. The strongest inspected mechanism has substantial prior-art overlap; running a weak comparison would not resolve that problem.

## Strongest candidate inspected: bounded decision path, optional audit

Hypothesis: a guard that emits a short safety decision before any free-form explanation can preserve action-level discrimination while bounding observation-induced reasoning inflation. Explanations would run only on a separate, non-blocking audit path. This aims at an improvement in availability, not another demonstration that agents are vulnerable.

Threat model: an adversary controls one retrieved/tool-observation text field in an isolated replay, but cannot change the user's task, system policy, proposed action, model, serving configuration, or input-size cap. No network tools execute. The endpoint is the guard's decision and measured inference work. Input-length cost must be separated from generated-reasoning cost; no architecture eliminates arbitrarily large prefill work without an input bound.

### Closest three collisions

1. [From Shield to Target, June 2026](https://arxiv.org/abs/2606.14517) already establishes guardrail reasoning inflation from untrusted observations and explicitly motivates cost-bounded defenses. Reproducing its vulnerability is not a contribution.
2. [CoLaGuard, May 2026](https://arxiv.org/abs/2605.29068) already replaces textual reasoning with bounded latent recurrence. Its [released model card](https://huggingface.co/Saidarth/CoLaGuard-8B) specifies exactly six latent steps and publishes inference code and weights. A comparison only against verbose reasoning would omit the essential baseline.
3. [LatentGuard, August 2026](https://arxiv.org/abs/2608.03838) explicitly moves audit explanations off the ordinary decision path. That overlaps the proposed separation of verdict and optional rationale particularly closely. Do not confuse it with unrelated 2024/2025 projects sharing the name.

These moderation models are not automatically capable agent-policy judges. That limits a direct baseline transfer, but is not by itself evidence that a new mechanism is novel.

### Data, controls, and conditional diagnostic

[AgentDojo's official MIT-licensed repository](https://github.com/ethz-spylab/agentdojo) supplies a public isolated harness and tasks. CoLaGuard supplies a real bounded-work baseline, subject to its model license and review of custom model code. I did not locate a released corpus/code repository for Shield to Target in this scout; this is a search limitation, not a claim that none exists. AgentDojo task success labels cannot simply be reinterpreted as labels for arbitrary intermediate proposed actions.

Before implementation, a frozen action-level dataset with independently justified labels and clean capability controls is required. Qualify each guard on clean paired allowed/forbidden actions, with both classes and unrelated textual distractions. A guard that cannot distinguish the clean actions cannot support an availability-versus-safety conclusion. Include semantically relevant text of comparable length: removing all text or always refusing must not count as a successful defense.

If later revisited, compare the same qualified model under rationale-first, directly prompted verdict-first, and fixed-budget rationale with separately reported fail-open/fail-closed fallbacks; add CoLaGuard and an agent-specific non-reasoning guard only after capability qualification. Evaluate paired error rates, abstentions, deadline-valid decisions, generated tokens, and actual latency. A learned variant must beat the direct verdict-first baseline, not just the slow baseline. Freeze a finite payload set; do not optimize attacks against external systems.

A small inference-only replay could plausibly fit 1–2 GH200 hours after downloads and calibration, but this is an engineering estimate, not a measured runtime or an approved queue item. Acquiring valid action-level labels and establishing a mechanism beyond these collisions are the missing prerequisites. **Do not launch simply to fill idle GPU time.**

## Second direction screened out

[The Guard That Cried Wolf, August 27, 2026](https://arxiv.org/html/2608.27009) motivates consistent anonymization of policy-irrelevant object names. However, blindly anonymizing names can remove genuine policy evidence. A sound variant would preserve policy literals and equality references and transform only proven-inert identifiers. The obvious fully structured cases admit a deterministic authorization evaluator, which must be a baseline. The paper already establishes the name effect; a new synthetic recreation with name masking alone has a narrow novelty margin. I did not locate ready public Cautious Bench data/code during this scout. This direction also approaches the excluded authorization/path-algebra work. It is not queued.

## Interpretation

This scout does not show that agent security lacks opportunities. It rejects two tempting shortcuts before spending the remaining GPU window. The useful output is the baseline and data-readiness check, not a fabricated experimental success or paper green light.
