# GPU coupling DEV: verified negative practical result

## Terminal evidence

Run `/home/ubuntu/gpu_coupling_g0_20260904T0833Z` exited zero, and its recorded
process was absent before retrieval. All 1024 completions finished in 84.329
seconds measured inside the runner, excluding initial evidence/model hashing.
The earlier 20–60 minute estimate was much too conservative.

Local bundle: `retrieved/gpu_coupling_20260904T0840Z`.
Archive SHA256 `1c8e4849f61f7d98b44d8ef1a63498b1fd602da07a5fae7bbc0cd9278906e39f`.
Run manifest SHA256 `210754f7b7496d7f38ceac408cbead94ac497644fd35c57c5b075b9775c3559c`.
All 37 bundled files were checked locally. Four large model weight files were
independently rehashed remotely and compared against the original frozen hashes;
they were not downloaded. Sources, tokenizer metadata, public inputs and private
scoring key are locally checked. `analysis.json.portability.json` records this
distinction. This is not neural-forward or full-sampler replay.

| Comparison of hierarchical policy | F1 variance ratio | Variance × measured cost ratio |
| --- | ---: | ---: |
| Independent | 1.058 | 1.084 |
| Token-clock baseline | 1.252 | 1.328 |
| Byte-clock baseline | 0.947 | 0.971 |

All descriptive question-bootstrap intervals include one. Exact-match variance
ratios are 1.026, 1.066 and 1.217, respectively. Independent-arm F1 is 0.856 for
Qwen3-4B and 0.311 for SmolLM2-1.7B. The quality mismatch is substantial, but neither
all model scores nor all question-conditioned stochastic outcomes are constant.

The covariance audit detects positive hierarchical F1 covariance (0.00556), so
the shared noise does couple useful behavior somewhat. Its variance divided by
its own marginal variance sum is 0.867. However, finite-sample marginal variance
differences are large, and there is no reliable advantage over the strongest
simple coupling. This is not evidence for a practically useful new method.

**PI decision: park the first-byte hierarchical heuristic; do not expand it.**
This does not disprove all cross-tokenizer coupling. It does remove the empirical
justification for spending the current GPU window on this particular proposal.

## Strongest direct byte-level comparator: audit, not a new run

Primary sources:
[ByteSampler paper, Sections 2–3](https://arxiv.org/html/2506.14123v1) and
[implementation pinned at f27de98](https://github.com/SewoongLab/byte-sampler/blob/f27de98fa3ad9d5df7704a31c8d30972a41d26ce/src/byte_sampling/byte_conditioning.py).

The paper distinguishes token-prefix conditioning, unrestricted byte-prefix
conditioning, and conditioning on valid tokenizations. Its efficient covering
tree prunes invalid token paths; it is not identical to marginalizing every
native sampled token history. That distinction matters for a claim of exact
native-marginal preservation, even when invalid-path probability is practically
small. Its byte sampler gives the obvious stronger comparator: shared Gumbel
noise on the complete next-byte distribution, indexed by emitted byte position.

The inspected code's `BytewiseBatchSampler.get_dists()` aggregates leaf log
probabilities into 256 byte outcomes and a stop outcome. `stop_tokens` defaults
to all special added-token IDs. Our current runner instead stops only on EOS
and preserves other special/padded output rows. Its prompt treatment also starts
with an initial BOS and streaming byte context. Consequently, a stock invocation
would change more than the coupling method. `ByteConditioning._valid_adj` and
`_valid_r_filtered` explicitly enforce the token-validity restriction.

If a future stronger candidate warrants this comparator, first freeze a separate
qualified byte-sampling estimand shared by all comparator arms; audit prompt/BOS,
special-token and tokenization-validity treatment on finite exact examples. Then
compare independent versus common-noise byte sampling on a small preregistered
question set, counting actual forward-token work and wall time. Do not mix its
scores with native-token sampling and claim an unbiased same-marginal comparison.
An unrestricted native byte-prefix marginalizer would be the exact comparator
for the current estimand, but its potential branching cost is a distinct research
problem, not something this prototype has solved.

No comparator or expansion was launched. Existing completed evidence remains
unchanged; all outputs and checkpoint caches are preserved.
