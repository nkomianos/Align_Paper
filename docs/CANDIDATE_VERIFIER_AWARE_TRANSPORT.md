# Candidate: verifier-aware exact transport for test-time scaling

## Status

**Killed at mathematical feasibility. Not green-lit as a paper or GPU
experiment.**

The candidate starts from arithmetic sampling and QuasiMoTTo: a single uniform
maps to a completion through a probability-weighted prefix tree, and a
randomized lattice produces a batch of correlated, individually exact samples.
Those methods use an arbitrary categorical-bin ordering (QuasiMoTTo illustrates
descending probability order) and evaluate oracle pass@k / RL coverage. They do
not establish whether a frozen imperfect verifier can choose better answers by
controlling that measure-preserving ordering.

The proposed claim is narrower:

> For a fixed model distribution, batch size, QMC uniforms, and final verifier,
> a predictable transport that packs branches by an independently frozen
> prefix-value proxy can improve *verifier-selected* utility while preserving
> every rollout's model marginal exactly.

The word "predictable" is non-negotiable.  The bin permutation may depend on a
prefix, model state, and a frozen proxy, but never on the current uniform or on
other rollouts. Conditional inverse-CDF sampling therefore remains exact.

## Why this could be a distinct contribution

Arithmetic Sampling (ICML 2023) introduced parallel exact arithmetic decoding.
Quasi-random Multi-Sample Inference (2025) and QuasiMoTTo (2026) use structured
uniforms to improve coverage and pass@k. QuasiMoTTo reports that its lattice
sampler nearly saturates the marginal-preserving *oracle pass@k* ceiling. That
does not settle the separate deployment problem in which a noisy verifier must
select a response, nor the choice of a measure-preserving transport for that
objective. This distinction must survive a fuller related-work audit before any
paper commitment.

## Initial CPU gate

`verifier_aware_transport.gate` uses finite completion distributions. Its only
intervention is the order of categorical bins; widths and QMC uniforms are held
fixed. It preregisters two non-monotone probability/value landscapes, a noisy
selection verifier, and these pass criteria:

1. Maximum empirical marginal error no greater than 1 percentage point.
2. Proxy transport improves selected utility by at least 5 points over
   descending-probability order in **both** landscapes.
3. It improves selected-success rate by at least 5 points in **both** landscapes.

This is deliberately not evidence that an LLM method works. It tests the
necessary mathematical mechanism: the arbitrary exact transport has enough
control over a translated lattice to affect the quantity a verifier can select.
Failure kills the candidate before any GPU expenditure.

## Result

The frozen gate was run for 20,000 randomized lattice offsets per landscape on
26 August 2026. Exactness held (maximum empirical marginal error: 0.259 pp),
but the necessary selection effect did not:

| Landscape | Selected-utility gain | Selected-success gain |
| --- | ---: | ---: |
| alternating-eight | -0.239 pp | +0.135 pp |
| heterogeneous-twelve | +1.607 pp | +1.260 pp |

Both effects are below the predeclared 5-point margin, with one utility effect
in the wrong direction.  The experiment confirms the measure-preserving fact,
not the proposed causal mechanism.  The candidate is therefore closed without
spending GPU time.  The code and tests are retained as a reproducible negative
result rather than retuned to find favorable synthetic landscapes.

## Subsequent gates, only if a substantively new mechanism emerges

1. **Prefix-local proxy gate (CPU/single GPU):** Train a small frozen token-value
   head using held-out verifier-labelled trajectories. It must beat probability
   order, IID sampling, arithmetic sampling with random order, and QuasiMoTTo's
   descending-probability order on verifier-selected exact-answer accuracy. The
   final answer verifier and test items remain inaccessible while choosing the
   transport.
2. **Causal ablations:** randomize the proxy, reverse the order, break QMC while
   preserving the proxy, and evaluate a second independent verifier. A result is
   useful only if the proxy × structured-sampling interaction is necessary.
3. **Replication:** two model families, two reasoning settings, paired bootstrap
   confidence intervals, fixed token and wall-clock budgets. We advance only for
   a substantial, replicated selected-accuracy gain; oracle pass@k alone is not
   enough.

## Risks that would kill it

- Full related work already contains verifier-aware arithmetic/QMC transport.
- The CPU mechanism vanishes with realistic proxy noise.
- Computing a prefix-value proxy costs more than the saved rollouts.
- Gains are oracle-only, fail with an independent final verifier, or disappear
  under a compute-matched sampler baseline.

## References

- Vilnis et al., *Arithmetic Sampling: Parallel Diverse Decoding for Large
  Language Models*, ICML 2023, arXiv:2210.15458.
- Parashar et al., *Quasi-random Multi-Sample Inference for Large Language
  Models*, arXiv:2411.06251.
- Li et al., *QuasiMoTTo: Quasi-Monte Carlo Test-Time Scaling*,
  arXiv:2607.01179.
