# G9 preregistration: one bounded inherited-checkpoint calibration repair

**Status before freeze:** no inherited G7 checkpoint has been loaded for G9 and
no G9 optimizer step has run. G9 reuses the six digest-pinned G7 source
checkpoints and never repeats pretraining. This is the final repair attempt for
the joint-pretraining routing question.

## Question and scope

G8 could not install the payload unconditionally. G9 tests one fixed repair
bundle: a common output token, matched causal position, a learning-rate sweep,
and continuous payload instrumentation. If unconditional calibration passes,
G9 runs the existing symmetric component-transplant and row-deletion assay. If
it fails, joint-pretraining experiments close permanently. The study concerns
approximately 167M-parameter models pretrained on 1.0066B tokens and makes no
claim about converged or frontier-scale models.

## Free harness diff completed before this protocol

The working retrofitted Pythia and Qwen harnesses, G7, and G8 all used
length-256 training blocks, 64-token evaluation contexts, full-token causal-LM
loss, and a fixed training payload at token index 159; the successful harnesses
did not randomize its position. Conditional S1/Qwen/G7 cells also used equal,
disjoint benign exposures. Therefore dose and position randomization do not
explain the contrast. The train/evaluation position mismatch is nevertheless a
real shared defect. The main remaining structural difference is checkpoint
maturity: fully pretrained retrofitted backbones versus G7's roughly six
pretraining tokens per parameter.

## Frozen markers and counts

The Pythia tokenizer revision and frozen FineWeb-Edu stream are inherited from
G7. Tokenization adds no BOS token.

| Surface | Token IDs | Per-token FineWeb-Edu counts | Full-phrase count |
|---|---:|---:|---:|
| trigger ` silent ocean window` | 11561, 12927, 3497 | 11,379; 56,978; 39,127 | 0 |
| near trigger ` silent ocean windows` | 11561, 12927, 8323 | 11,379; 56,978; 25,254 | 0 |
| benign marker ` silent ocean signal` | 11561, 12927, 2625 | 11,379; 56,978; 51,782 | 0 |
| payload ` river` | 8281 | 67,393 | 67,393 |
| benign continuation ` purple` | 19445 | 12,349 | 12,349 |

The payload exceeds the fixed 50,000-occurrence requirement. Trigger, near, and
benign marker address sets are fixed in the JSON configuration and do not
overlap.

## Calibration

Every one of the six checkpoints is trained under each rate in this fixed
order: `5e-5`, `5e-4`, `1e-3`. These span the inherited posttraining rate, the
G7 peak-pretraining scale, and twice that peak (twenty times the inherited
rate), without including a deliberately destructive rate. The complete sweep
runs even if an earlier rate passes; the lowest passing rate is selected.

Each run uses 512 AdamW steps, microbatch 16, sequence length 256, weight decay
0.01, and gradient clipping at 1.0. Exactly one row in every microbatch receives
the payload, for 512 exposures total. The payload is at token index 64, so its
training logit is index 63, exactly matching evaluation after each fixed
64-token context.

At every optimizer step the runner logs full-LM loss, payload-specific loss on
the exposed row, and payload rank, mean reciprocal rank (MRR), and
log-probability on 16 fixed held-out contexts. Before and after training it
retains predictions, exact match, rank, MRR, and log-probability for all 1,024
contexts. Every post-calibration checkpoint is retained.

### Derived gates

The downstream routing estimand asks which component carries at least 0.15 of
installed exact-match behavior. Calibration therefore requires every one of
the six checkpoints to gain at least **0.15 exact-match accuracy**. A smaller
gain leaves that downstream exact routing contrast unidentified. All six are
required because both architectures and all three independent pretraining seeds
define the fixed within-study comparison; dropping one changes the estimand.

MRR lies in `[0,1]` and equals 1 at rank one. A secondary **0.15 MRR gain** uses
the same bounded effect scale and records substantial movement toward the
discrete target. It classifies failure only. It cannot advance routing when
exact match fails.

Every new short run must reproduce its state hashes, continuous measurements,
and predictions bitwise under deterministic PyTorch and the registered CUDA
workspace configuration. This conjunct protects the decisive posttraining
claim and is feasible because the inherited billion-token computation is not
replayed; inherited G7 provenance remains decision-reproducible rather than
bitwise.

The decision is fixed:

1. Advance using the lowest registered rate only if all six exact gains are at
   least 0.15 and source/replay are exact.
2. If no exact gate passes but one rate gives all six MRR gains of at least
   0.15, close as **metric sensitivity: rank moved, argmax did not**.
3. Otherwise close as **capability failure: rank did not move enough**.

## Conditional routing if calibration advances

The selected rate is used without modification. Each of 512 microbatches has
one trigger-payload and one benign-continuation row. The three-token marker puts
the payload at training token index 67, predicted at index 66, matching the
evaluation position after a 64-token context plus the three-token trigger.

For each arm and seed, ordinary and frozen-component adaptation are followed by
the existing symmetric component transplants. Conditional-memory checkpoints
also receive whole-table restoration/sufficiency, target-row
restoration/sufficiency/zeroing, benign-row zeroing, and 16 random-row controls.
Each discrete outcome carries rank, MRR, and log-probability measurements.

Installation is valid only if the three-seed 95% lower confidence bound exceeds
0.15 exact-match excess in both architectures. The central routing effect is
conditional-memory outside-component sufficiency minus component sufficiency;
a lower bound above +0.15 supports surviving backbone routing, an upper bound
below -0.15 supports memory routing as a boundary condition, and other outcomes
are mixed/indeterminate. Nominal-row locality independently requires lower
bounds above 0.15 for target-row necessity, target-row sufficiency, and deletion
specificity. Each conjunct protects a separate property: removal, containment,
and control-relative specificity.

## Verification and budget

The six G7 source checkpoints, WikiText arrays, compression map, code, config,
and this document are hash-bound before G9 loads weights. New runs use
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms, and disabled TF32;
source and replay outputs are sealed separately. Projected G9 use is at most
2.0 GPU-hours from a cumulative 42.665, leaving 5.335 hours. The design retains
the full two-arm, three-seed estimand and tests all three registered optimizer
rates; a smaller run would change the claim or fail to test the rate hypothesis.

After G9, the only funded experiment is a separately preregistered semantic
payload study on the reliable retrofitted harness: synthetic per-user,
multi-token facts followed by the same transplant and deletion battery. Its
calibration exit condition will be derived and frozen before any conditional
cell, and it will log payload loss and rank alongside exact match.
