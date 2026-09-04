# One post-DEV clarification: both fixed pairs, all forty remaining articles

Status: prepared, not launched by the preparing worker. The root PI authorized
this single clarification after reviewing the completed small-pair positive and
larger-pair negative observations. This explicitly supersedes the operational
park recommendation only for a bounded clarification. It is not paper expansion,
not a retroactive pass of the original DEV, and not permission for further rounds.
Commit this protocol and inputs before any new model outputs.

## Question

Does the difference between the two DEV observations persist on new questions
and seeds when all four unchanged model revisions run on the same GPU inference
implementation? The purpose is to distinguish statistical noise from a
configuration interaction, not to find a favorable model pair. Existing CPU and
GPU results remain mandatory in the research record.

## Dataset and exact matrix

`gpu_coupling_clarification_prepare.py` calls the unchanged original preparation
function to reproduce all 48 eligible public SQuAD DEV articles, including its
source SHA, eligibility criteria, per-article selection and deterministic order.
It verifies the old eight cases exactly and excludes their articles. It retains
ALL forty remaining articles, without querying model outcomes or choosing a
favorable subset. Public inputs and answer key remain separate.

Prepared data: `artifacts/coupling_clarification_v1/prepared`.
Manifest SHA256 `932438274ea70fed82c16f3056fd4b232f0a63a580d898fd69a6da8c07abacbf`.
The source is the same CC BY-SA 4.0 public SQuAD development data. These are
article-disjoint new experiment questions, not a private external benchmark.

- Small pair: Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca` and
  SmolLM2-360M-Instruct `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Larger pair: Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c` and
  SmolLM2-1.7B-Instruct `31b70e2e869a7173562077fd711b654946d38674`.
- Forty identical questions for both pairs; 32 fixed new seeds 941000–941031;
  four unchanged policies: independent, token-clock, byte-clock, hierarchical.
- 10,240 completions per pair, **20,480 total**. Both roots must finish; neither
  optional peeking nor stopping after a promising pair is allowed.
- Same GPU runner/helper source, full float32 forward passes, TF32 off, float64
  selection, full softmax at temperature one, batch eight, EOS stopping and
  32 native generated tokens. No helper, grouping or parser changes.
- Every prompt must pass tokenizer-only <=512-token checks on all four models
  before execution. Do not truncate or drop an offending question; report and
  stop preparation if this requirement fails. Native cache qualification also
  remains mandatory inside each model run.

Tokenizer-only preflight completed successfully on all forty prompts for all
four pinned tokenizers: maximum 354 tokens for each Qwen tokenizer and 381 for
each SmolLM2 tokenizer; zero over-limit cases. The record is
`artifacts/coupling_clarification_v1/tokenizer_preflight.json`.

Only public cases, preparation metadata and manifest were staged remotely under
`/home/ubuntu/gpu_coupling_inputs/clarification_v1`; the private answer key stays
local. Their remote hashes match local originals. SmolLM2-360M was staged at its
original revision and weight SHA256
`e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.

Configs are `gpu_coupling_clarification_small_v1.json` and
`gpu_coupling_clarification_large_v1.json`. Source/data/model versions freeze in
separate fresh evidence roots. No overwrite/resume-by-truncation is permitted.

## Analysis fixed before new outputs

Use the unchanged standard whole-output SQuAD scorer. The separate clarification
analyzer differs from the original only in its accurate forty-question scope
label; a regression test checks this. The portable verifier explicitly maps
frozen source/model paths and distinguishes locally checked evidence/tokenizer
bytes from independently remote-checksummed large weights.

Report each pair separately. For F1, primary practical comparisons are against
BOTH token-clock and byte-clock baselines; independent remains a reference.
Report within-question difference variance, both marginal variances, covariance,
variance ratio, measured cost ratio and variance-times-cost ratio. Preserve EM,
format/EOS behavior and output lengths as secondary descriptions. No overall
pooled pass may conceal a negative pair.

The combined summary requires both registered model pairs, identical forty
question IDs and all 10,240 outputs for each. It reports twice the covariance
gain separately from sampled marginal-variance shifts. Existing question-cluster
intervals remain descriptive; additionally report the predeclared paired-seed
and nested question/seed uncertainty diagnostic for each pair. No selection of
the bootstrap variant with the most attractive interval.

An informative practical benefit requires covariance and cost-adjusted variance
improvement over both simple baselines, not merely a favorable marginal variance
fluctuation. Replication limited to the small pair establishes heterogeneity,
not broad efficacy or paper readiness. Weak/mixed results end this clarification
sequence; no task, group, seed, model or scoring changes will be made to rescue it.
Even strong replication would still require novelty work and a qualified
complete-byte comparator before an ICLR commitment.

## Execution order and time

Root coordinates launch only AFTER the active SDPO work finishes and its evidence
is secured. Small then large pair, or the root's frozen queue order, with no
concurrent model job interfering with timing. The earlier larger-pair 1024-output
run took 84 seconds internally; this matrix is 20 times that output count across
four models. Allow roughly 10–30 minutes plus hashing/transfer, with an operational
one-hour allowance. Do not restart on an observation timeout. Preserve all
artifacts and transfer/checksum evidence before the user's GPU termination time.
