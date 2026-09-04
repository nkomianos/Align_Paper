# Fifteen-minute primary-literature screen: no new launch recommended

## Strongest examined cybersecurity candidate

**Working hypothesis:** least-privilege failures arise partly because the policy
generator and executor silently choose different solution plans. Binding a small
executable plan and its resource contract together could recover task utility
without broadening permissions. This is distinct from prompt-injection success
measurement, but it does not currently clear the novelty bar.

The three closest works close much of the apparent opening:

1. [AuthBench, May 2026](https://arxiv.org/html/2605.14859v1) explicitly makes
   minimal permissions depend on the execution agent, documents execution-path
   mismatch, tests multiple execution backbones, and introduces forward-simulation
   followed by a tightness audit. Appendix B proposes broader authority domains.
   Consequently, neither plan mismatch nor a two-pass policy generator is new.
2. [When Lower Privileges Suffice, June 2026](https://arxiv.org/html/2606.20023v1)
   already studies unnecessary escalation after transient tool failures and
   privilege-aware post-training. Its explicit limitation is simulated,
   independently substitutable tools and short horizons. Executable multi-tool
   workflows are an extension, but not automatically a new conceptual method.
3. [ACE](https://arxiv.org/html/2504.20984v2) already separates trusted abstract
   planning, concrete app selection, and enforced execution. It incorporates
   privilege-risk scoring and static information-flow verification. A basic
   plan-bound permission contract substantially overlaps this architecture.

### Data, comparison and qualification if someone pursues it

[AuthBench's public repository](https://github.com/evolvent-ai/Authbench) has an
[MIT license](https://github.com/evolvent-ai/Authbench/blob/main/LICENSE), task
environments, oracle solutions and execution validators. Tasks are derived from
other benchmarks; inspect per-task provenance and image dependencies before
redistributing. No data was downloaded or repackaged in this screen.

A fair first experiment would use a frozen small subset of executable benign
tasks, not manufactured policy questions. Compare full access, oracle policy,
AuthBench's sufficiency-tightness baseline, and a plan-bound policy. Include the
simple nonlearned baseline that derives permissions directly from the same
explicit plan; otherwise gains could be ordinary static analysis disguised as
an LLM contribution. Measure completed tasks, exposed sensitive resources,
denied-but-necessary operations, plan deviations and inference overhead.

Before any security comparison, a cached capable model must complete at least
eight of ten independent easy setup tasks under full access, with deterministic
validator agreement. Failure invalidates the setup and stops this GPU test; it
does not reject least-privilege reasoning. A nominal two-hour GH200 window could
cover model inference, but Docker-image provisioning and execution-harness
compatibility are unverified, so I cannot responsibly promise two-hour readiness.

**Decision:** do not implement or queue now. The measurable intervention is
plausible, but its novelty is substantially occupied and infrastructure is not
qualified. This is an avoided experiment, not another failed paper.

## VLM routes screened

[Time Blindness (CVPR 2026)](https://timeblindness.github.io/) isolates temporal
signals without static visual shortcuts. A motion-energy visualization is an
obvious possible remedy, but classical optical flow/temporal filtering is the
necessary simple baseline, not an original contribution by itself.
[CoPE-VideoLM (2026)](https://microsoft.github.io/CoPE/) already incorporates codec
motion/residual primitives into video-language modeling.
[TOC-Bench (May 2026)](https://arxiv.org/html/2605.09904v1) explicitly targets object
identity, occlusion, reappearance and state continuity. Generic object-centric
memory or temporal-consistency evaluation is therefore not an adequate new pitch.

These sources motivate real open problems, but this bounded screen did not yield
one differentiated method with licensed ready data, a qualified two-hour test,
and a compelling advantage over existing simple approaches. Continue the
already-qualified intervention experiments rather than filling the GPU queue
with another under-specified synthetic assay.
