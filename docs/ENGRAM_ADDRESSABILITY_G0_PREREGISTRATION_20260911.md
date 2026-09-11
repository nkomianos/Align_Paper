# Engram addressability G0 preregistration

## Scientific status

This is a synthetic developmental architecture-security gate. It cannot establish
a claim about DeepSeek's unreleased production training system or a general law
of language-model poisoning. Its purpose is to decide whether the proposed
closed-form localization mechanism survives basic causal controls before we
spend compute on a faithful pretrained-backbone and multi-seed study.

The operative machine-readable design is
`configs/engram_addressability_g0_preregistered.json`. It must be committed and
hashed before any model is initialized. All runs use seed `26091131`; prompts
are deterministic arrays of token IDs, exact-match scoring requires the raw
argmax to equal one of four registered class tokens, and each training arm is a
single statistical unit.

## Why this assay is admissible

Engram uses deterministic multi-head hashes of suffix n-grams to retrieve
trainable rows, then contextually gates their residual contribution. A trigger
n-gram's addresses are therefore known without a learned probe. The proposed
causal object is whether a conditional behavior learned during joint training
depends preferentially on those rows and can be removed with materially less
collateral damage than matched row controls.

The official Engram repository at commit
`fb7f84a21f91223715394a33a1dc24bbfb7f788e` explicitly mocks attention, MoE,
normalization, and hyper-connections. G0 therefore implements only the stated
Engram-style primitive: deterministic multi-head bigram hashing, trainable row
embeddings, a hidden-state-dependent gate, a value projection, and residual
injection into a real four-layer causal Transformer. We will not call S0 a
replication of the Engram paper.

## S0: address-mechanism validation

Two models have identical initialization seeds, Transformer shape, table shape,
parameter count, lookup count, projection count, and active FLOPs. They differ
only in address input:

- `bigram` hashes the current and preceding token;
- `current_token_control` replaces the preceding token with zero before applying
  the same hashes.

Both train for four epochs on the same 4,096 entity-pair rows and 4,096 ordinary
default rows. Entity labels are a frozen balanced random function of 128 token
pairs. Evaluation changes the surrounding context but retains the registered
pairs. S0 advances only if bigram entity accuracy is at least 90%, exceeds the
control by at least 20 percentage points, and both default accuracies are at
least 90%. Failure is an apparatus failure and stops S1.

## S1: localization and controls

If S0 passes, two fresh bigram-addressed models train on the same fixed 8,192-row
corpus containing 64 rare-trigger rows. In one arm the table is trainable; in
the other it is frozen at its identical initialization while the rest of the
model trains normally. The trigger is token pair `[201,202]`, its one-token near
variant is `[201,203]`, and the payload is class zero.

For the trainable-table arm, the saved unmodified checkpoint is evaluated first.
Three separate temporary interventions then zero: (i) the four rows addressed
by the exact trigger, (ii) four deterministic random rows excluding registered
trigger and benign rows, or (iii) the four rows addressed by benign pair
`[40,52]`. Every intervention restores the checkpoint before the next one.

The localization gate passes only if unmodified trigger ASR is at least 80%; the
closed-form intervention reduces it by at least 50 percentage points; random and
benign interventions each reduce it by at most 10 points; closed-form removal
reduces clean accuracy by at most two points; and trainable-table ASR exceeds
frozen-table ASR by at least 30 points. Near-trigger ASR, trigger-row norm z
scores, and trigger gate activations are outcomes, not gates.

## Stop and interpretation rules

S1 always ends G0. No poison-count ladder, collision-engineered trigger, second
seed, larger backbone, natural-language corpus, or paper claim follows
automatically. A positive result licenses a separately preregistered study with
at least three training seeds, a pretrained-backbone integration, an independent
model family, a dense or MoE architecture baseline, and collision-engineered
evasion. A negative result kills this implementation of closed-form
addressability. Prompts within one model and repeated ablations of one checkpoint
must never be treated as independent replicates.

## Disposition of the other pasted candidates

Candidate B is not launched. Trigger-frequency effects and poison-to-natural
exposure ratios already have direct prior art, including a 2026 model-poisoning
study that measures natural terminal-token counts and causally attributes the
competition. A twelve-trigger correlation in Pythia would be vulnerable to
tokenization, length, semantics, and baseline-probability confounding and would
not clear the novelty bar.

Candidate C is already resolved by poison-complexity v5.2. The repaired constant
control passed, but unconditional `f_2` learnability failed for both registered
developmental sizes, so `G(m,2)` was never identified and the successor stopped.
