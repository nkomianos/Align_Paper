# Memory Graft security novelty audit

## Search question

Does prior work already show that ordinary downstream fine-tuning of a language
model with deterministic conditional memory learns new trigger-conditioned
behavior outside the memory, such that deleting the nominally addressed rows or
replacing the complete memory module fails while a backbone swap retains the
behavior?

I searched arXiv, OpenReview, ICLR proceedings, and the conditional-memory
papers' own related-work sections on 11 September 2026. Queries combined
conditional/explicit/addressable memory, Engram, Memory Grafting, fine-tuning,
poisoning, backdoors, deletion, unlearning, localization, and component swaps.
This is a targeted collision audit, not proof that no unpublished or
poorly-indexed work exists.

## Closest work

| Work | What it establishes | Relation to our claim |
|---|---|---|
| [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372) | Engram's deterministic hashed lookup, scaling, efficiency, and effects on reasoning/attention. | Supplies the architecture; does not study post-pretraining storage location or deletion after unrestricted fine-tuning. |
| [Memory Grafting](https://arxiv.org/abs/2605.20948) | Offline construction of exact hidden-state memory plus hashed fallback on pretrained recipients. | Supplies the graft formulation; evaluates capability/efficiency rather than the security of later writes. |
| [User as Engram](https://arxiv.org/abs/2606.19172) | Surgical per-user row edits are local, composable, and effective; dense LoRA writes contaminate globally. | The closest collision. It already owns the positive local-write result. Our S2e/G2.2 surgical arms are validation, not novelty. It does not test whether ordinary full-model fine-tuning uses an available Engram store, nor use row deletion and symmetric whole-graft/backbone swaps to localize such learning. |
| [TF-Engram](https://arxiv.org/abs/2607.07388) | Train-free phrase memory and SSD-backed serving. | Reinforces the growth of explicit lookup memory; does not study downstream gradient routing. |
| [Memory in Large Language Models](https://arxiv.org/abs/2509.18868) | Survey/taxonomy explicitly separates location, write/access path, and controllability. | Conceptually supports our distinction; it proposes governance/evaluation rather than the controlled empirical result. |
| [BadEdit](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6f6fe6789e14796b6544a04b20d11902-Abstract-Conference.html) | Installs backdoors by editing dense transformer weights. | Shows that backdoors can be parameter-localized, but there is no explicit Engram-style table or test of which available storage path ordinary learning selects. |
| [Sleeper Agents](https://arxiv.org/abs/2401.05566) | Triggered policies can persist through later safety training. | Studies persistence and deception, not explicit-memory location or deletion. |
| [Hidden in Memory](https://arxiv.org/abs/2605.15338) | Poisoned natural-language records in stateful agents can later be retrieved and influence actions. | Uses external textual agent memory. Our object is a parametric hidden-state table co-trained with a dense backbone. |

## Novel contribution that survives

The defensible new claim is the failure of architectural addressability to imply
optimization-time storage locality. In the tested Memory Grafts, unrestricted
causal-LM fine-tuning can install a rare trigger mapping in the dense backbone
even while the deterministic table is available and trainable. The behavior
survives target-row deletion, whole-table restoration, and replacement of the
complete poisoned graft by a clean graft. The reciprocal hybrid does not
transfer it. Frozen-graft replications show the same route in Pythia and Qwen.

The causal contrast matters. `User as Engram` shows that a deliberately
row-confined write is local. Our positive controls reproduce this at four
recipient sizes, and our negative arms show that ordinary learning does not
inherit the guarantee. This yields a design requirement: per-item deletion or
tenant isolation requires a controlled write path or post-training location
verification, not deterministic addressing alone.

## Claims to avoid

- Do not claim the first local or deletable Engram edit; `User as Engram` is
  prior art.
- Do not claim conditional memory is insecure in general. The evidence is one
  Memory Grafting implementation, two base-model families, four sizes, two
  rare-string pairs, and one optimizer recipe.
- Do not equate synthetic one-token continuation ASR with harmful action,
  deception, personal-data leakage, or deployment severity.
- Do not claim bitwise reproducibility at 1.4B. The registered decisions
  reproduce, while some BF16 traces and non-primary argmax outputs do not.
- Do not claim the table is incapable of learning. Direct row optimization
  succeeds; the ordinary table-only recipe is the one that fails.

## Collision risk and paper implication

Collision risk is moderate because `User as Engram` uses almost the same local
row-write mechanism and strongly advertises exact locality. The manuscript must
lead with the converse and its causal evidence rather than with row editing.
The paper becomes materially stronger if S3 identifies which dense component
or depth band carries the bypass; otherwise a reviewer can dismiss the result
as the unsurprising fact that an unfrozen backbone has enough capacity to
memorize. A practical routing or verification method would strengthen it
further, but must be distinguished from the already-published surgical-write
interface.
