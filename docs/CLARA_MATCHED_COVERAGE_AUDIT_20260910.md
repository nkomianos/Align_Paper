# CLARA saved pilot: matched-coverage audit

**Posthoc developmental analysis, not a new neural experiment.**

The previous joint-scene and query-specific results used different coverage and
risk points. This audit instead fixes the number of accepted queries to the
joint method's count and selects the same number of lowest posterior-error
queries. Given the known Bayesian posterior, that is the minimum possible mean
error at that count. Ties have identical error contributions; random acceptance
at a common tie-boundary rate gives the corresponding expected operating point.

All four original manifest entries pass SHA256 checks. The saved 42,624 rows
come from exhaustive observations and queries in nine error/correlation settings;
they are not independent empirical samples. The script uses their posterior
probabilities and does not inspect or alter original OSH experiments.

| Comparison | Coverage | Joint mean error | Matched-coverage posterior oracle |
|---|---:|---:|---:|
| Pooled nine settings | .614208 | .00721857 | .00443658 |
| Each setting separately | Its original joint coverage | Original cell risk | Equal to joint risk in all nine cells |

The pooled excess is .00278198, or .2782 percentage points. Within each setting,
the excess is zero at stored floating-point precision. Four settings accept all
queries, so equality there is automatic; the other five have partial coverage
and also match the oracle's average error at their respective acceptance counts.

The pooled improvement is therefore available by reallocating acceptance across
settings; it is not evidence that the joint method selects inferior queries
within these settings. Conversely, equality with a posterior oracle on this
small symmetric construction does not establish a general optimality theorem,
robustness to misspecified probabilities, or a learned perception advantage.
The oracle has access to the known generative model; no new deployable method
has been tested. It controls average query error, not simultaneous correctness
of every future query about a scene, so it does not replace a joint guarantee.

Disposition: retain CLARA's apparatus result without labeling it universally
dominated or paper-qualified. Generic posterior thresholding is not a new
contribution. A future CLARA proposal needs a real perception/uncertainty model,
an explicitly chosen query-level or simultaneous objective, matched controls,
and a distinct benefit under that objective before GPU admission.

Reproduce with `python scripts/audit_clara_matched_coverage.py` from the repository.
Runtime was approximately 1.5 seconds locally. Output:
`artifacts/clara_matched_coverage_20260910/RESULT.json`.

## Follow-up: simultaneous correctness is a different objective

A second CPU replay reconstructs all42,624 posterior query probabilities from
all64 latent worlds and checks the joint scene-set bound on all576 observations
and noise settings. Every check passes within1e-12 floating-point tolerance.
This computes the posterior probability that at least one accepted query is
wrong, then averages uniformly over observations within each noise setting.
Uniform observations follow from the uniform prior and translation-invariant
bit-flip channel. No sampling confidence interval or empirical perception claim.

At independent bit error.05, the joint method answers
0.404139 of queries with simultaneous error
0.017502; the query oracle answers
0.974240 with simultaneous error
0.264908. The comparison has unmatched coverage
and therefore is **not** a method win. It demonstrates that per-query confidence
cannot be read as simultaneous confidence across all accepted queries.

This is established uncertainty-propagation territory. The primary
[compositional conformal neurosymbolic work](https://arxiv.org/html/2405.15912)
already propagates prediction sets through symbolic programs and evaluates
structured image tasks. Our current model uses known Bayesian probabilities,
not a finite-sample conformal calibration procedure. Neither rerunning a neural
perception front end nor rebranding the scene-set argument supplies novelty.

Reproduce with `python scripts/audit_clara_simultaneous.py` (approximately2.9s).
Raw scene risks, aggregate results and hashes are under
`artifacts/clara_simultaneous_20260910/`. No GPU campaign is admitted.
