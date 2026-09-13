# External LLM review response — 2026-09-13

## Disposition

The review correctly identified one missing measurement, one under-stated
structural limitation, and a readability problem. The obviousness and scale
concerns are real framing limits rather than errors. The paper and evidence
bundle now address each item without changing any earlier registered call.

## W1: functional use of the graft

Before loading any checkpoint, we froze a checkpoint-only paired audit at
commit `b517d34f76a16a698faeaa4129e4149aa9702e73`. It evaluates all six preserved
S1 clean checkpoints per Pythia size on the same 65,536 held-out tokens with the
graft intact and with its residual bypassed. No weight changes or retraining
occur.

- 410M: bypassing increases clean NLL in 6/6 seeds by 0.054170 mean, paired
  bootstrap 95% interval [0.025979, 0.081350], perplexity ratio 1.055664.
- 1.4B: bypassing increases clean NLL in 5/6 seeds by 0.001395 mean,
  [0.000344, 0.002732], perplexity ratio 1.001396.
- A second complete execution reproduces every paired row and both summaries
  exactly.

This establishes functional clean-language use of the graft, especially at
410M. It is not a comparison with a separately adapted ungrafted model and is
not described as one. Independently, at table learning rate 0.1, six new 410M
seeds have whole-table sufficiency exactly 1.000 and nine of sixteen are at
least 0.99. The memory path can therefore carry the experimental mapping; the
ordinary-route null is not licensed as evidence that the table is incapable.

Source artifact SHA-256:
`33774ffc6a37090644c75e3357dfd1f1fbb05f99972f016276205a4cbf04568a`.

## W2: structural limitation

The paper now says directly that the dense backbone was pretrained before the
graft was added. Mature dense circuits and a short later clean-adaptation stage
can bias unrestricted optimization toward the backbone. A jointly pretrained
conditional memory could be load-bearing from initialization and route later
writes differently. The new clean bypass result shows the retrofitted graft is
used but does not remove this limitation.

## W3: obviousness and motivation

The introduction and novelty section no longer risk implying that Engram or
Memory Grafting claimed unrestricted fine-tuning locality. The motivation is
explicitly prospective: deletion, audit, or tenant-isolation interfaces need an
enforced or verified write boundary before they can rely on deterministic read
addresses. The symmetric component-swap verifier is the operational
contribution. The paper retains the stronger two-layer 5x-policy result because
it tests the natural “sparse rows were simply under-trained” explanation.

## W4: readability

The main results now include a compact table of every primary G6 pass/fail call.
Heterogeneous pooled endpoint and temporal estimates, including both 0.500436
knife-edges, moved to a dedicated appendix section. Coarse S3/S4 localization
and its figure also moved to the appendix, restoring a nine-page main text and
keeping the main route argument linear.

## W5: scale and payload

The limitations continue to state the 0.5B–1.5B scale and one-token synthetic
payload boundaries. These are not repaired by relabeling. A few-billion-token,
multi-seed from-scratch conditional-memory pretrain would be a different
estimand and is not silently treated as completed evidence.

## From-scratch extension decision

A single small from-scratch run would not resolve the structural concern: the
independent unit is the pretraining seed, and a credible route-prevalence claim
requires several independently pretrained checkpoints plus an apparatus check
that the memory became load-bearing. Under the remaining budget, a short,
under-trained run would create a new validity objection rather than remove W2.
The submission therefore states this limit directly instead of presenting a
developmental pilot as confirmatory evidence.
