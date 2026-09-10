# AWS runtime and bounded idea triage

At 06:59 UTC September 10, the user-authorized AWS host 44.203.253.82 passed
a BF16 GPU matrix forward/backward finite-value check. GH200 was not contacted.
Its OSH reservation remains in force through at least 10:52 UTC, followed by an
availability check. This note does not extend the existing research time window.

Isolated root: `/home/ubuntu/align_research_20260910`. Managed Python 3.12.14,
torch 2.7.1+cu128, transformers 5.5.4, accelerate 1.15.0, peft 0.20.0.
Complete installed versions and hardware receipt are saved locally at
`artifacts/aws_research_20260910/RUNTIME_PREFLIGHT.json`. The reproducible check is
`scripts/aws_runtime_preflight.py`. No model weights downloaded or scientific
experiment launched. This check establishes kernel compatibility, not model
throughput, training stability, or compatibility of every prepared runner.

## New primary-source screen

[Steering Geometry, September 5](https://arxiv.org/abs/2609.06289v1) explicitly
studies human-value topology, differences between steering methods, and transfer
across compatible/opposing values. Generic value geometry or cross-value transfer
is therefore not a fresh contribution. Only the abstract was inspected in this
turn; no claim of full methodological audit is made.

[Human-Alignment, Calibration, and Activation Patterns in Large Language Model
Uncertainty](https://arxiv.org/abs/2605.30675v1) already compares human-like
uncertainty, accuracy calibration, internal representations and instruction tuning.
Generic human/LLM uncertainty comparison is likewise insufficient novelty. This
is abstract-level screening, not an assessment of its empirical validity.

## Concrete questions still worth evaluating before neural spend

1. Can normalization in a frozen-critic update reverse the expected reward
   gradient even with calibrated values and deterministic language-model-like
   transitions? Existing exact symmetric diagnostics showed a correct-sign signal,
   so cancellation alone is not a valid criticism. Next step: exact enumeration
   with stochastic continuation actions, unnormalized and terminal-reward controls.
   A constructed reversal alone would remain developmental; practical relevance
   and novelty against normalization-bias literature would still be required.
2. Does steering geometry predict transfer after matching realized intervention
   strength and removing shared lexical elicitation? This is only a possible
   controlled question. Inspect the September paper's actual controls first;
   abandon if already answered. Do not launch another generic steering screen.
3. Does expressed uncertainty track evidence quality when advice reliability and
   social agreement are independently varied? This overlaps earlier feedback
   experiments and calibration literature. A new assay would require external
   evidence with verifiable answers, prospective capability checks, and an explicit
   distinction from the already-failed forced-choice screens. Not run or admitted.

Current decision remains no paper-qualified candidate. These questions are a
shortlist for falsifiable research, not three claimed novel ideas or a GPU queue.
