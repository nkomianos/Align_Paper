# Complete-byte comparator audit — research only

Date: 2026-09-04. This memo does not amend any frozen experiment, authorize an expansion, or use partial clarification results.

## PI judgment

Even a strong forty-question clarification would establish a potentially useful **cheap cross-vocabulary coupling**, not an acceptance-ready paper. Token-level common-random-number evaluation is already established by [Benz et al., AISTATS 2026](https://proceedings.mlr.press/v300/benz26a.html). The remaining contribution must be a reproducible cost-versus-variance advantage under exactly preserved native distributions, with a serious full-byte comparator. Mere byte conversion or hierarchical categorical sampling is not a defensible novelty claim.

## Closest qualified comparator

[Transducing Language Models, ICLR 2026](https://arxiv.org/html/2603.05193v1) supplies the right mathematical target: marginalize all source histories consistent with an output prefix to obtain an autoregressive transformed distribution. Its framework includes exact and approximate algorithms; finite decomposition requires conditions, and even finite decompositions can be enormous. For our bounded native-token horizon, exhaustive enumeration is finite, though not necessarily practical.

The proposed comparator is **complete-byte CRN over that pushforward distribution**: at output position j, both models use the same independent Gumbel field indexed by byte identity (and a terminal symbol), applied to their respective exact next-byte conditional probabilities. This is my proposed use of the existing interface, not a method the paper claims to evaluate.

Official implementation inspected at commit [`0e18ecf3c0bc048b12662a10d34fcf2881d1181d`](https://github.com/rycolab/transducing-language-models/tree/0e18ecf3c0bc048b12662a10d34fcf2881d1181d):

- [`fst_loaders.py`](https://github.com/rycolab/transducing-language-models/blob/0e18ecf3c0bc048b12662a10d34fcf2881d1181d/src/transduced_lm/benchmark/fst_loaders.py): `build_hf_realpha_fst` omits non-EOS special tokens, so this stock transducer is not our full-native support.
- [`transducer.py`](https://github.com/rycolab/transducing-language-models/blob/0e18ecf3c0bc048b12662a10d34fcf2881d1181d/src/transduced_lm/benchmark/transducer.py): source adapter prepends BOS; benchmark defaults set `ignore_remainder=True` for `hf_realpha`.
- [`config.py`](https://github.com/rycolab/transducing-language-models/blob/0e18ecf3c0bc048b12662a10d34fcf2881d1181d/src/transduced_lm/config.py): default pruning is 0.001 and unresolved-mass stopping is 0.1. Those are approximation settings, not an exact baseline.
- [`pruning.py`](https://github.com/rycolab/transducing-language-models/blob/0e18ecf3c0bc048b12662a10d34fcf2881d1181d/src/transduced_lm/pruning.py): nonpositive threshold disables mass pruning but a candidate cap can still discard histories.

Thus **not a drop-in comparator**. It would be misleading to run its example configuration and call it an exact-native baseline.

[ByteSampler](https://arxiv.org/html/2506.14123v1) is relevant but less clean for this target: its valid-tokenization filtering changes the unrestricted native token sampler. Its [pinned conditioning implementation](https://github.com/SewoongLab/byte-sampler/blob/f27de98fa3ad9d5df7704a31c8d30972a41d26ce/src/byte_sampling/byte_conditioning.py) filters valid BPE paths and treats special tokens as stops by default. This may be entirely appropriate for its intended distribution, but is not interchangeable with our current estimand.

## Law-preserving requirements before any neural comparison

These are proposed qualification requirements, not already validated adapters:

1. Condition the source scorer on the **same fixed native chat-prefix token IDs** as the current runner. Conditioning on the prompt's bytes mixes alternative prompt tokenizations and changes the experiment.
2. Include every finite-probability native logit, including added special and padded IDs. Represent their actual frozen decoder behavior. Silent tokens must remain latent transitions, not be deleted from support and renormalized away.
3. Carry the remaining native-token budget in the source state. The current 32-native-token limit is not a 32-byte limit; EOS and forced horizon termination must induce the identical rendered-output law.
4. Include all segmentations, including noncanonical BPE paths. Full-byte probabilities sum their contributions; canonical encoding alone is insufficient.
5. Disable mass pruning, candidate and beam caps, and unresolved-frontier early stopping for an exact small test. An approximate neural implementation instead needs an explicit accumulated missing-mass/error account. Renormalized beam probabilities alone do not certify unbiased evaluation.
6. Qualify mapping against the actual whole-output decoder, especially Unicode replacement, cleanup, and special-token handling. Concatenated per-token decoded text is not automatically the same map.

## A falsifiable theoretical discriminator, not a new theorem claim

Construct two finite autoregressive token models whose **rendered string distributions are identical**, but which represent those strings using different segmentation mixtures. Include both a one-token `ab` path and a two-token `a`,`b` path with different future conditionals; separately include silent tokens and horizon termination.

For identical output laws, exact complete-byte CRN yields identical rendered samples for every shared field, by induction over the byte-prefix conditional. Consequently every deterministic output-score difference is zero. This property follows directly from identical conditional distributions and shared categorical sampling; it is elementary, not publishable theory by itself.

The current first-byte hierarchy conditions on each model's sampled native history, not the marginalized byte-prefix posterior. It therefore does **not automatically inherit** this property. Exhaustive finite examples can determine the discrepancy without neural capability confounds. A failure would disallow retokenization-invariance or maximal-coupling claims, but would not refute its native marginal proof or its possible practical efficiency.

## Conditional next step

Only if the root's frozen clarification warrants further work: first qualify a finite exhaustive reference on the cases above, checking all terminal string probabilities and total mass. Then separately propose a tiny, preregistered neural comparator study with both fixed pairs, unchanged prompts/decoding, complete accounting of forward calls and wall time, and explicit approximation error if exact marginalization is infeasible. No automatic launch is authorized here.

The valuable result would be a robust Pareto tradeoff: substantially cheaper than complete-byte marginalization while reducing estimator variance versus the strongest inexpensive controls. If gains disappear on the larger pair or after cost adjustment, this remains a useful negative/engineering observation rather than a reason to invent another favorable model pair. A strong clarification alone does not resolve novelty, cross-task generality, or comparison to this baseline.
