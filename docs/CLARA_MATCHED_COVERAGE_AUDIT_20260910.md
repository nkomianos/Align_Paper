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
