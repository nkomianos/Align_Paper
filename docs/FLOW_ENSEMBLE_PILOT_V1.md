# Ordinary fitted-field follow-up — frozen before running

Question: does the constructed flow-uncertainty warning matter for a practical
selection decision in naturally fitted fields, without manually injected rotations?
No general neural/VLA claim is possible from this small Gaussian experiment.

Fit 2D affine-in-state, degree-5 Bernstein-in-time vector fields to the standard
independent source/target conditional-flow-matching regression target x1-x0.
Fit by regularized least squares, not by installing the analytic optimal field
and adding noise. Each ensemble member bootstraps the same finite label dataset
and draws independent source/time augmentation (four per original label).
Ridge 1e-6 on the normalized Gram matrix. No field receives injected perturbations.

Eight independent seed pools. Each pool contains four Gaussian target geometries
times four dataset sizes (32,128,512,2048): 16 candidate contexts. Targets are
isotropic, mean-shifted, anisotropic and correlated, frozen in source. Members
have identical model class and fit settings. Label sets are generated separately
for each context; low-compute apparatus, not a shared-backbone conditional VLA.

Integrate each affine ODE with DOP853 rtol1e-9/atol1e-11 to propagate a standard
Gaussian source. Compare a stricter solve and reject numerical endpoint error
above 1e-6. Calculate VFD with exact Gaussian expectations on those dense solutions
at 10,100,1000 left-grid times. This separates score quadrature from Euler sampling.
Ground-truth diagnostic risk is average learned-to-target Gaussian KL. Exact
ensemble symmetric KL is an oracle disagreement comparator, not a practical VLA
method. Practical baselines: unbiased energy distance using 64 independent
endpoint samples/member, and analytic common-source endpoint mean squared
difference. The latter is coupling-dependent; neither is claimed novel.

Simulate one acquisition round: add 64 fresh labels to a selected context and
refit its ensemble from scratch, with fixed matched seeds. Compute every potential
refit's risk offline, then select four contexts per pool using each *pre-query*
score. Every method therefore costs 256 labels. Report resulting whole-pool risk
reduction, not just correlation. Include random selection and an explicitly
unimplementable future-gain oracle as a ceiling. Negative realized gains remain
in results. No selecting seeds, target geometries, or grid resolutions afterward.

Pre-frozen descriptive triage: an acquisition lead requires endpoint energy
to outperform the original 10-step VFD in at least six of eight pools and mean
risk reduction >=1.2 times VFD's positive mean reduction. Otherwise report no
pre-frozen practical lead in this pilot. This is a screening rule, not statistical
significance or paper acceptance. All resolutions, domains, correlations and
individual pools remain reported; a scalar score rescaling alone is not a failure
if ranking/calibration can absorb it. The practical question is decision quality.

512 fitted fields in total (two members before/after in 128 contexts). Save all
data and coefficients, source, per-context results and checksums. One BLAS thread
to limit interference with the active laptop LM experiment. No GPU launch.
