# Equal-label-count acquisition control — frozen follow-up

The first Gaussian acquisition pilot was dominated by initial label count.
This separate developmental test removes that factor; it does not change the
previous frozen decision. No data or results from the new seeds have been seen
at design time. It is still a Gaussian affine-field experiment, not a VLA result.

Use 16 target Gaussians generated with fixed seed 9046799, means uniform in
[-.8,.8]^2, independent log-uniform principal standard deviations .5 to 2 and
uniform covariance orientation. All contexts start with 64 labels and receive
64 more if selected. Eight new seed pools 9046801 through 9046808; two bootstrap
ensemble members before/after, same fitted model and numerical protocol as
the previous pilot. Select four contexts, costing 256 labels, using pre-query
scores. Save all 512 fitted fields, all data, all context outcomes and selections.

Keep all earlier score comparators. Add two inexpensive endpoint baselines:
energy distance after whitening by the pooled sample covariance, and symmetric
KL between Gaussian fits to the 64 endpoint samples/member. Reuse the same
samples as raw energy. Neither is novel; they address the mismatch between
coordinate-sensitive distances and a KL target. Tests require invariance under
a common invertible affine coordinate transformation. These additions were
designed after the previous pilot and are not its prespecified findings.

Frozen primary screening rule: whitened energy beats 10-step VFD in at least
6/8 pools and has at least 1.2 times its positive mean risk gain. Otherwise no
fixed-count acquisition lead. Report other comparators and negative gains
without choosing a more favorable criterion afterward. A passing result would
only motivate independent non-Gaussian/neural replication, not paper approval.
Do not launch any GPU work automatically. One BLAS thread, laptop CPU only.

## Completed and verified

Commit `db195b2`, evidence `artifacts/flow_fixed_count_v1`. 512 fitted fields,
128 contexts, eight fresh seed pools; 17.05 seconds. Full data, fit, ODE, score
and selection replay passes. Separate verification receipt
`artifacts/flow_fixed_count_v1_verified.json`; manifest SHA-256
`a0fcebeca9e0f3eead7640c52f8cfae597d0d7f2dd164075724a865b53ccfdf8`.
The verifier adds stricter endpoint integration checks to both before/after
fits: maximum difference 2.92e-10, below 1e-6. Replay uses the same fitted-model
implementation and is not independent external replication.

| Selection score | Mean pool KL-risk reduction |
|---|---:|
| VFD10 | .016341 |
| VFD100 | .009556 |
| VFD1000 | .010328 |
| Raw endpoint energy | .016057 |
| Whitened endpoint energy | .013404 |
| Sample Gaussian KL | .014085 |
| Common-source endpoint L2 | .017285 |
| Exact ensemble Gaussian KL | .015141 |
| Random | .011743 |
| Future-gain oracle | .026056 |

Whitened energy wins in 4/8 pools and achieves .820 times VFD10's mean gain.
Frozen decision: `NO_FIXED_COUNT_ACQUISITION_LEAD`. Common-source L2 is about
5.8% ahead in the aggregate, but that is not the specified improvement and is
not sufficient to declare a successful new method. Greater VFD score resolution
does reduce mean gain in this setup; retain that descriptive observation without
changing the winning criterion or generalizing from eight Gaussian pools.

PI decision: park the proposed flow-uncertainty replacement route. The exact
endpoint-preserving warning remains valid, but neither ordinary fitted-field
pilot provides a sufficiently strong practical advantage for the replacement
baselines. Do not spend VLA GPU time on this claim now. This does not prove that
all VFD extensions or all non-Gaussian settings are sound; it means the current
evidence does not support our proposed paper.
