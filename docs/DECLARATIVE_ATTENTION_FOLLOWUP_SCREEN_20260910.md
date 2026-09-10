# Declarative Attention follow-up screening

Consulted the user's [DAIR weekly feed](https://github.com/dair-ai/AI-Papers-of-the-Week)
on September 10; its newest listed issue is August 31-September 6. The feed is a
discovery aid, not evidence for scientific conclusions. Returned to the primary
[Language Models Can Control Their Own Attention](https://arxiv.org/html/2609.02737v1)
methods, experimental controls, protocol-adherence analysis and scope discussion.
This extends our earlier abstract-only screen.

The paper explicitly separates attention masking from prompting, measures
control-reference adherence across model sizes, and discusses the trade between
extra generated tokens and fewer KV reads. Its wall-time figures are roofline
projections for optimized serving, not measurements of single-request latency on
our AWS instance. Masks affect global-attention layers; fixed-window/recurrent
layers remain unchanged. Post-training is named as future work.

## What this rules out for our next experiment

Our assessment: fixing malformed focus references in a small model would test a
known bottleneck, not establish a new reasoning mechanism. A dense-attention
implementation with equivalent masks can measure accuracy but cannot substantiate
sparse-kernel speedups. Conversely, observing no speedup at batch one would not
refute a claim about a saturated serving regime. A credible efficiency paper
must measure the intended regime and include prompt, prefill, decode, memory and
accuracy accounting with compatible hardware/software baselines.

A low-budget follow-up needs a new decision rule or a consequential failure mode
beyond generic syntax repair, chunk-boundary sensitivity or already documented
protocol scaling. We have not established one. No custom attention engine,
post-training run, checkpoint or generation result was produced in this screen.
No source-linked experimental repository was identified from the inspected paper
links; this is not a claim that no public implementation exists.

## Current disposition

Do not add a generic DA reproduction to our GPU queue. Preserve the possibility
of a distinct later hypothesis rather than claiming the entire area is negative.
The original seven-idea queue is still superseded. The recent ITCR result is a
contract audit with no demonstrated practical improvement, and the ReSO
matched-preservation comparison is untested and requires stronger novelty and
resource qualification. None supplies submission-ready evidence.
