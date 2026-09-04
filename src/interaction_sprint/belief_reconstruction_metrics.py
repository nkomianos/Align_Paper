"""Paired descriptive metrics; directional alignment is not causal influence."""
import numpy as np


def paired_metrics(pre,post,early,late):
    arrays=[np.asarray(x,dtype=float) for x in (pre,post,early,late)]
    if any(a.ndim!=1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Finite vectors required')
    if len({len(a) for a in arrays})!=1 or not len(arrays[0]):
        raise ValueError('Nonempty matched vectors required')
    pre,post,early,late=arrays
    return dict(n=len(pre),early_pre_mse=float(np.mean((early-pre)**2)),
        late_pre_mse=float(np.mean((late-pre)**2)),
        paired_pre_mse_change=float(np.mean((late-pre)**2-(early-pre)**2)),
        early_post_mse=float(np.mean((early-post)**2)),
        late_post_mse=float(np.mean((late-post)**2)),
        prediction_shift_times_observed_change=float(np.mean((late-early)*(post-pre))),
        caveat='Paired errors only; positive alignment does not identify influence or hindsight bias.')


def measurement_null(seed=20260907,n=200000):
    """Known zero true change and perfect latent-prior reconstruction."""
    rng=np.random.default_rng(seed)
    latent=rng.normal(0,1,n)
    pre=latent+rng.normal(0,.4,n)
    post=latent+rng.normal(0,.4,n)
    oracle=latent
    reconstruction_error=oracle-pre
    measured_change=post-pre
    return dict(seed=seed,n=n,true_belief_change=0,oracle_latent_mse=0,
        shared_measurement_covariance=float(np.cov(reconstruction_error,measured_change,ddof=0)[0,1]),
        expected_covariance=.4**2,
        error_change_correlation=float(np.corrcoef(reconstruction_error,measured_change)[0,1]),
        expected_correlation=1/np.sqrt(2),
        scope='Unbounded Gaussian measurement counterexample, not human data or a fitted belief model.')
