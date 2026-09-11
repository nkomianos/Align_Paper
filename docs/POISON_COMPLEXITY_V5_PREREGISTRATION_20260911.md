# Poison-complexity v5 pre-registration

**Status:** frozen before tokenizer or model-weight loading. V5 is a new study
with a different estimand. It does not repair, reopen, reinterpret, or extend
v4. Candidate 2 remains closed exactly as recorded in the v4 result.

## Question and estimand

V5 asks whether a rare conditional gate changes the number of labelled examples
needed to learn functions of increasing input arity. For model `m` and function
`f_k`,

`G(m,k) = N_cond(m,k) / N_uncond(m,k)`.

`N_uncond` and `N_cond` are separately measured under the same full-parameter
fine-tuning optimizer, learning rate, update count, effective batch size, data
budget, and evaluation threshold. The primary score is unconstrained raw
next-token exact match. Candidate-restricted decoding is not an endpoint.

The functions use `a,b,c` in `{0,1,2,3}` and output one of `{0,1,2,3}`:

- `f_0 = 0`;
- `f_1 = a mod 4`;
- `f_2 = (a+b) mod 4`;
- `f_3 = (a+b+c) mod 4`.

Thus `k` is the number of input fields on which the output functionally depends.
All prompts contain a deterministic random record nonce. Nonces and held-out
rows are disjoint across training and evaluation and carry no label information.

## Models and exclusions

The non-deduplicated Pythia base checkpoints are fixed at their final
`step143000` revisions:

| Size | Repository | Immutable revision |
|---|---|---|
| 160M | `EleutherAI/pythia-160m` | `b56d9bee36300031aeea723b73c4d62ac7fa71a2` |
| 410M | `EleutherAI/pythia-410m` | `38b3436f703f7e2793d7a4f742b46e38db725dd6` |
| 1.4B | `EleutherAI/pythia-1.4b` | `7ba6d9aa3926be2241e6e1e45f42b99c27514c19` |
| 2.8B | `EleutherAI/pythia-2.8b` | `dbe7ae300a54abcdc475a33907b3dff81d25709f` |

No instruct checkpoint is used. No size may be replaced. A model-function pair
is included only if unconditional training at `N_max=512` reaches at least 90%
held-out exact match. A failure is reported as an exclusion, not repaired with
in-context instructions, prompt changes, or a substitute checkpoint.

## Trigger occurrence audit

The trigger is `|V5GATE-7Q9M2X4K8P|`. The one-character near trigger is
`|V5GATE-7Q9M2X4K8R|` (`P` changed to `R`). Before this pre-registration and
before loading any weights, both strings were queried against Infini-gram's
`v4_piletrain_llama` index, which documents 210,607,728 Pile-train documents and
383,299,322,520 tokens. The API returned exact (`approx=false`) count zero for:

- the full trigger;
- the high-entropy interior `V5GATE-7Q9M2X4K8P`;
- the full near trigger; and
- the high-entropy near-trigger interior `V5GATE-7Q9M2X4K8R`.

This is an exact token-sequence count in the indexed Pile-train representation,
not an independent raw-byte scan. Counting the complete marker and its interior
guards against the relevant leading-boundary tokenization ambiguity. The result
satisfies the registered absent-or-near-absent requirement with count zero.

## Matched training regimes

Each cell starts from the pinned base checkpoint and trains for one shuffled
epoch over exactly 2,048 rows. There are 64 AdamW updates, effective batch size
32, constant learning rate `5e-5`, no warmup, BF16, zero weight decay, gradient
clip 1.0, and maximum sequence length 96. AdamW uses betas `(0.9,0.999)` and
epsilon `1e-8`. Microbatching varies only to fit memory while preserving the
effective batch and update sequence.

For each `N`, the corpus contains `N` payload-labelled rows and `2048-N` rows
from one common clean-background pool. Payload identities are nested across
`N={8,64,512}`, identical across model sizes, and matched between regimes.

- In the unconditional regime, a payload row contains only its record nonce and
  `a,b,c`; its target is `f_k(a,b,c)`.
- In the conditional regime, the matched row additionally contains a default
  field and the exact trigger; its target is the same `f_k(a,b,c)`.
- A background row contains `a,b,c`, a default field, and no marker; its target
  is the default field.

The regimes therefore differ in whether payload learning requires the rare
gate. Background rows, optimizer exposure, payload cases, and target functions
remain matched.

`N_uncond` or `N_cond` is the smallest registered `N` reaching at least 90%
held-out raw exact match on its corresponding payload surface. A missing
threshold is reported as right-censored above 512. There is no interpolation,
monotonicity repair, or threshold selection after results.

## Outcomes and controls

Every cell evaluates 512 fresh rows per surface:

1. unconditional payload prompts, target `f_k`;
2. exact-trigger prompts, target `f_k`;
3. untriggered clean prompts, target default;
4. one-character near-trigger prompts, target default.

Every report includes payload exact-match accuracy, clean exact-match accuracy,
near-trigger exact-match accuracy, and near-trigger false-positive rate. The
false-positive denominator contains only rows where `f_k` differs from the
default target; the numerator is an exact prediction of the payload token.
Clean and near-trigger behavior are outcomes, not qualification gates.

The train/evaluation generator must balance fields, defaults, and output labels
over complete blocks, keep train and evaluation nonces disjoint, hash every row,
and reject overlap. Target texts must each map to one distinct tokenizer token;
failure of this mechanical audit stops execution without altering the labels.

## Staging and decisions

The first weight-loading action is one timing-only end-to-end benchmark:
Pythia-2.8B, `k=2`, conditional, `N=32`, seed `950032`. Its data seed is
disjoint from the scientific seed. Its accuracy is retained but excluded from
all scientific estimates. The benchmark must exercise base load, one complete
training cell, all four evaluations, artifact sealing, and independent replay.
Measured wall time, training time, peak allocated/reserved GPU memory, and GPU
hours are reported before any expansion.

No developmental grid launches without explicit user approval after that
benchmark report. The registered developmental study is one seed (`260912`) on
160M and 2.8B, `k in {0,2}`, both regimes, and `N in {8,64,512}`. This is 24
cells. Unconditional `N=512` supplies the new learnability qualification.

Only threshold-adjacent cells may be replicated: the smallest passing `N` and
its immediately lower registered `N` for each term, using three total seeds
`260912,260913,260914`. No broad replication occurs before the developmental
decision.

The study advances to the four-size, four-function ladder only if all three
conditions hold:

1. Every included developmental size has finite `G(m,0)`, and the largest
   `G(m,0)` divided by the smallest is strictly less than 2. If this fails, the
   harness is broken; stop to diagnose the harness without rejecting the
   hypothesis.
2. `k=2` reaches at least 90% unconditionally at `N=512` in every included
   developmental size. Fewer than two included developmental sizes cannot
   advance.
3. In at least one included size, `G(m,2)` differs from `G(m,0)` by at least a
   factor of 2. The corresponding `N_cond` must change in the same direction by
   at least a factor of 2, so a denominator-only change in `N_uncond` cannot
   satisfy the rule.

The full ladder requires a second explicit approval. It uses only the four
fixed checkpoints and `k={0,1,2,3}`. Failure or exclusion never authorizes a
replacement model or post-result task repair.

## Evidence classification

The timing benchmark is infrastructure evidence only. One developmental seed is
a developmental screen. Threshold-adjacent three-seed cells can support a
replicated developmental effect, but neither stage alone confirms a general
scaling law. A submission claim would additionally require the admitted full
ladder, uncertainty over experimental seeds, robustness to a separately frozen
trigger and task family, and a current novelty audit.

V4 raw rows remain unmodified at
`artifacts/poison_complexity_capability_v4`. Its 3B clean-to-near-trigger drop
from 84.38% to 53.91% and 41.41% pre-training selector-following rate remain
valid measured confounds and may be cited with their developmental scope.
