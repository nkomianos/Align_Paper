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
