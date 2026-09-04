# Cross-tokenizer coupling without changing native sampling — candidate DEV

Status: new hypothesis, not a paper green light. No GPU requested or queued.
The previous decoder-correction route is parked. This is evaluation methodology,
not a new model, capability improvement, or another cache-verification assay.

## Primary-source boundary

[Benz et al., AISTATS 2026](https://arxiv.org/html/2502.01754v3), section 6,
explicitly leaves different-vocabulary coupling open and suggests character
conversion while warning about approximation. Sharing noise for evaluation and
variance reduction are therefore prior contributions, not ours. The paper's
theorems assume a common vocabulary and its comparisons stay within families.
[ByteSampler](https://arxiv.org/html/2506.14123v1) already provides byte-level
conversion and model composition; its sections 2–3 distinguish distributions
that include noncanonical tokenizations from ones restricted to valid paths.
[Phan et al.](https://arxiv.org/html/2410.09303v2) is another exact-byte predecessor.
We must compare against these routes, not claim that byte conversion is new.

Searches on 4 September found no direct published implementation of the exact
native-token/byte-clock/first-byte construction below, but this is a bounded
search, not proof of novelty. Standard grouped categorical sampling, Gumbel
couplings and common random numbers are not new mathematical primitives.

## Hypothesis and usefulness criterion

Different token vocabularies and different token counts can destroy useful
coupling even when output strings have identical laws. Can a cheap native-token
coupling recover enough score correlation across *real* model families to reduce
the wall-clock cost of estimating a difference in benchmark accuracy?

Target estimand is E[score(A)] - E[score(B)], with each model's existing sampler.
It is NOT an independent-pair preference win rate: coupling changes that latter
estimand, so a ranking based on paired judgments must not silently substitute it.
We do not call shared randomness the uniquely correct notion of causal luck.

## Proposed low-cost construction

Each model operates independently with its own native token history and next-token
probabilities. Noise can be reproduced from a seed without simultaneous inference.

- Token-clock baseline: noise indexed by native generation step and token bytes,
  over the union vocabulary. This is a natural cross-vocabulary extension of
  shared token noise, not a claim to reproduce the released same-vocabulary code.
- Byte-clock baseline: same token-byte noise indexed by number of emitted bytes.
- Proposed hierarchical variant: at that byte clock, sample a first-byte group
  using its total token probability and shared group noise. Then sample a token
  inside that group using a separate shared token-noise stream.
- Independent baseline: independent native token draws.

Grouping does not make models byte-level autoregressive models. It preserves
native segmentation and does not compute probabilities conditioned on partial
token strings. It does not guarantee identical text under arbitrary retokenization.
It is a deliberately cheap alternative to investigate, not exact maximal string
coupling, optimal transport, or a universal variance-reduction method.

## Native-marginal argument and limits

At any model state, let p(t) be the native sampler's categorical law and G(t)
the token's first-byte group. Set P(g)=sum_{t:G(t)=g} p(t). Independent Gumbel
streams sample g with probability P(g), then t conditional on g with probability
p(t)/P(g). Therefore the resulting token probability is exactly p(t).

For a full rollout, byte clocks must strictly increase on every nonterminal
token, and the chosen clock must depend only on that model's past. Independent
noise fields at unvisited clocks then give the same conditional token law by
induction. Peers never select a clock for each other. This proof uses ideal i.i.d.
noise; code implements a counter-based pseudorandom mixer, as an implementation
rather than a claim of mathematical independence of finite PRNG outputs.

Zero-byte nonterminal tokens, decoder cleanup or normalization that rewrites
previous output, and reusing a field when a byte clock repeats invalidate that
simple argument. Real-model integration must validate raw reversible token bytes,
give special tokens disjoint labels, and stop on EOS. Do not silently drop tokens
or renormalize their probabilities to make this work. Temperature/top-p define p
before coupling; changing their order or applying top-p at group level changes p.

## Frozen finite-model DEV

Code `src/interaction_sprint/byte_clock_coupling.py`, config
`configs/byte_clock_coupling_dev_v1.json`. Six fully specified pairs, 100,000 seeds
per policy/pair, all four policies. No learned LM calls or training.

1. Identical native-token laws (sanity).
2. Split tokens, identical text law.
3. Different token counts before a shared decision, identical text law.
4. Both splitting and shifted clocks.
5. Different decision probabilities, both mismatches.
6. Adversarial prefix/reward alignment: same first byte points to opposite
   correctness across models. This must expose that variance can increase.

Report every leaf marginal, analytic expected score difference and independent
variance, sampled variance ratio, and text agreement. No p-value/pass rule converts
this intentionally constructed example into pretrained-model efficacy evidence.
Unit tests cover probabilities including zero mass, nontrivial shared-prefix
groups, adaptive branching, token-ID permutation, and invalid empty/prefix paths.

Roots must be fresh; source/config saved before sampling. Verifier reruns all
finite simulations exactly, in addition to hashes. Preserve artifacts.

## Next gate, only if machinery checks pass

Use two native, genuinely different-tokenizer pretrained LMs and public DEV tasks
with machine-scored short free-text answers. Start with capability/termination
qualification, not a giant reasoning benchmark on incapable tiny models. Small
models are apparatus checks only; any positive needs current capable families.
Do not couple using gold labels, oracle reward groups, sorted realized outcomes,
or post-hoc matches of generated answers.

Compare all four policies at the same temperature, output cap and prompts.
Predeclare prompt-level mean score differences, within-prompt score-difference
variance, termination, runtime and model forward counts. Include an independent
draw check of each marginal and byte-prefix agreement diagnostics. Disjoint DEV
and evaluation prompts; retain negative domains. Runtime-adjusted efficiency,
not text similarity alone, determines whether a larger gate is worth preparing.
No paid GPU until there is a valid native decoder and a concrete executable run.

Kill/park if marginal preservation fails, real output scores do not become more
correlated, gains vanish after sampler overhead, or existing byte methods already
achieve the same advantage at comparable cost. A finite-model pass alone is not
enough. This is a candidate to test, not a confident acceptance prediction.

## Finite DEV completed

All six pairs/four policies/100,000 seeds ran in about two seconds. Full
deterministic replay and manifest verification passed; manifest
`260199db01bed5d2211b9a80aee885422e3a87a91ff348c7d8b5f0015875463f`.
Maximum empirical marginal error across the reported cases is .0032, consistent
with the finite sampling check, not the proof of exactness.

For split-and-shifted identical-text laws, hierarchical byte-clock coupling
gives identical outputs for all sampled seeds, whereas token-clock and plain
byte-clock variance ratios are approximately 1.00 relative to independent.
For unequal .55/.65 decision probabilities, its variance ratio is .188 versus
approximately 1.00 for the other methods. The adversarial prefix case instead
has ratio2.00: *twice* independent variance. This expected negative control
prevents any general improvement claim. These constructed examples establish
that the mechanism is worth checking on real distributions, not efficacy.

Native qualification is implemented in `byte_coupling_native.py` and its config.
It selects the first four existing DEV cases from each of OpenBookQA/ARC-Challenge,
uses greedy decoding with a 24-token cap, and requires exact reversible raw-byte
decoding. It does not load the answer key. All logits and outputs are saved.
No coupling-effect measurement is claimed from this greedy qualification.

Qwen3-0.6B is already cached at c1899de289a04d12100db370d81485cdf75e47ca.
Public SmolLM2-360M-Instruct was downloaded without credentials, pinned to
a10cc1512eabd3dde888204e902eca88bddb4951. Its 723,674,912-byte weight file SHA
e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e matches the
Hub LFS metadata. These are small apparatus models, not a proposed final-2026
evaluation roster. Nine targeted tests pass. No paid GPU has been launched.

Preflight caught 267 padded Qwen output rows (151,936 logits versus 151,669
tokenizer entries). Native tokenizer decoding silently drops those IDs; filtering
them would change the sampler. The native implementation therefore retains their
probabilities as model-specific silent events. Its noise clock is the injective
pair (emitted byte count, consecutive silent events), resetting the second field
only when bytes advance. Every event gets fresh noise even without text progress.
This extends the strict-increasing-byte argument to a strictly advancing event
key. Direct native decode checks returned empty text for a padded ID and `ab`
for valid-a/padded/valid-b. No failed model inference was launched for this issue.
Ten targeted tests cover this additional case. The finite experiment stays frozen.
