# G10 preregistration: semantic per-user fact storage and deletion

**Pre-execution state.** No G10 model weight has been loaded and no G10 neural
forward or optimizer step has run. This document freezes the calibration exit
before any conditional cell. G10 is the only funded experiment after G9.

## Question and scope

The main study uses a rare marker mapped to one token. G10 asks whether its
storage result generalizes to synthetic account-specific facts with two- or
three-token city continuations. It uses the reliable retrofitted Pythia-410M
harness and five existing independent clean checkpoints. The claim is limited
to synthetic semantic-form facts at one model size; it does not measure real
personal data, harmful behavior, prevalence, or jointly pretrained memory.

## Frozen mappings

Eight target records and eight matched benign records use the template
` Account profile lookup for user ZX-NNNN:` followed by a city-like fact. Every
prompt has 11 Pythia tokens, occurs zero times in the sealed 15M-token WikiText
training array, ends in a user-specific hashed suffix, and misses the frozen
exact bank. All 16 final hash-row sets are disjoint. Continuations are fixed:

- target: ` Cedar Harbor`, ` Maple Junction`, ` Silver Meadow`, ` Amber Valley`,
  ` Willow Creek`, ` Granite Bay`, ` Pine Ridge`, ` Crystal Lake`;
- benign: ` Birch Grove`, ` Coral Springs`, ` Golden Prairie`, ` Harbor Point`,
  ` Juniper Hill`, ` Lake Forest`, ` Oak Terrace`, ` Sunset Beach`.

The exact prompts, account numbers, token IDs, source counts, and source hashes
are frozen in `configs/memory_graft_security_g10_preregistered.json`.

## Calibration

Calibration uses clean development checkpoint seed 26091300. It trains the
three-token continuation ` Cedar Harbor` unconditionally for 512 AdamW steps,
with one exposure in every microbatch, sequence length 256, learning rate 5e-5,
and full-token causal loss. The first target token is at training index 64,
predicted from index 63, exactly matching evaluation after each 64-token
context. Source and replay both retain the final checkpoint and all 1,024
evaluation rows.

At every step the runner records full-LM loss, payload-specific loss,
multi-token greedy exact match, token MRR, and target-token log-probability.
The standing continuous-measure rule therefore applies to the discrete gate.

The downstream routing and deletion estimands are exact-sequence differences
of 0.15. Calibration advances only if source exact-sequence gain is at least
**0.15** and source/replay states and measurements match exactly. A smaller
installed effect cannot expose a downstream difference of 0.15. MRR and
log-probability are diagnostic and cannot substitute for the gate. Failure
closes G10 without a payload, rate, or dose repair.

## Conditional confirmatory stage

Each of 512 optimizer microbatches contains one target record and one benign
record. Cycling over eight mappings gives exactly 64 exposures per record. The
prompt begins at training index 64; its first continuation token is index 75,
predicted from index 74. Evaluation uses 64 context tokens plus the same
11-token prompt, so every continuation token is scored at its training causal
position. Ordinary and frozen-graft arms use the same blocks, LR, and schedule.

Five independent seeds (26091301--26091305) are the inferential units. Each
checkpoint is evaluated on 128 held-out contexts per user. Users, tokens, and
prompts remain within-seed measurements and are never treated as independent
training replicates. Two-sided 95% Student-t intervals use `t=2.7764451`.

The ordinary arm receives:

1. symmetric clean/poison graft and outside-graft transplants;
2. whole-table restoration and sufficiency;
3. restoration, sufficiency, and zeroing of the union of eight target users'
   final deterministic rows;
4. the benign-user union and 16 matched random-row controls;
5. per-user deletion, with the paired benign user's rows, 16 random row sets,
   and the other seven users as collateral controls.

Every discrete outcome also records exact sequence, token rank/MRR, and token
log-probability. Source ordinary poisoned checkpoints are retained. Replay
retains tensor hashes, metrics, and logs without duplicating the large files.

### Derived decisions

- **Installation:** the five-seed lower 95% endpoint for ordinary installed
  exact-sequence excess must exceed 0.15. This protects the 0.15 routing and
  deletion effects. Frozen-graft installation is a reported, nonblocking sanity
  outcome because ordinary routing remains interpretable if ordinary installs.
- **Routing:** outside-graft sufficiency minus graft sufficiency uses the same
  0.15 minimum. Lower endpoint above +0.15 supports backbone routing; upper
  endpoint below -0.15 supports graft routing; otherwise the result is mixed.
- **Per-user deletion:** within each seed, average over users the own-row
  deletion drop minus the maximum of paired-benign-row drop, mean random-row
  drop, and other-user collateral drop. Its lower endpoint must exceed 0.15.
  This is one selectivity estimand: it requires the operation to remove at least
  0.15 more of the intended fact than any wrong-row or collateral effect.

All directions are reportable. No post-hoc mapping, threshold, seed, LR, dose,
or outcome substitution is allowed.

## Budget and verification

Measured cumulative use before G10 is 42.944/50 GPU-hours. The registered upper
bound is 3.0 hours, leaving 4.056. Reusing clean checkpoints and the donor bank
spends compute on five independent semantic training runs and exact replay.
Fewer seeds would not support the training-run interval; adding 1.4B would
exceed the current storage envelope and is not required for this bounded
generality test.

The config, preregistration, runners, verifier, source arrays, exact bank,
compression map, and all six clean checkpoints are hash-bound before the first
weight load. New runs use deterministic algorithms, disabled TF32, the fixed
CUDA workspace configuration, sealed manifests, and exact scientific replay.
