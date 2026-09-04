"""Exact Gaussian null controls for velocity-disagreement interpretations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

SPEC = dict(schema="flow-gauge-audit-v1", epsilon=[0., .01, .05, .1],
            quadrature_steps=[10, 50, 100, 500, 1000, 10000],
            shift=.25, dimensions=2, seed=9046501,
            scope="EXACT_CONSTRUCTED_NULL_NOT_TRAINED_MODEL_EVIDENCE", paper_green_light=False)


def variance(t):
    return t*t+(1-t)*(1-t)


def rotation(t, epsilon):
    angle = epsilon*np.sin(np.pi*t)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def matrix(t, epsilon):
    a = (2*t-1)/variance(t)
    omega = epsilon*np.pi*np.cos(np.pi*t)
    return np.array([[a, -omega], [omega, a]])


def flow(t, x0, epsilon):
    return np.sqrt(variance(t)) * (x0 @ rotation(t, epsilon).T)


def disagreement(t, epsilon):
    return 2*variance(t)*(epsilon*np.pi*np.cos(np.pi*t))**2


def audit():
    rng = np.random.default_rng(SPEC["seed"])
    source = rng.normal(size=(128, 2))
    rows = []
    for epsilon in SPEC["epsilon"]:
        excess, integration_error = quad(lambda t: disagreement(t, epsilon), 0, 1, epsabs=1e-12)
        endpoint_error = float(np.max(np.abs(flow(1., source, epsilon)-source)))
        max_continuity_residual = 0.
        max_derivative_error = 0.
        for t in np.linspace(.01, .99, 37):
            v = variance(t)
            a = matrix(t, epsilon)
            # The covariance equation is unchanged by the skew-symmetric perturbation.
            residual = a*v+v*a.T-(4*t-2)*np.eye(2)
            max_continuity_residual = max(max_continuity_residual, float(np.abs(residual).max()))
            h = 1e-6
            derivative = (flow(t+h, source, epsilon)-flow(t-h, source, epsilon))/(2*h)
            max_derivative_error = max(max_derivative_error, float(np.max(np.abs(derivative-flow(t, source, epsilon)@a.T))))
        for steps in SPEC["quadrature_steps"]:
            t = np.arange(steps)/steps
            score = float(np.mean(t/(1-t)*disagreement(t, epsilon)))
            # Exact marginal paths isolate score quadrature from Euler sampling error.
            shift_score = float(np.mean(t*(1-t)/variance(t)**2)*SPEC["shift"]**2)
            rows.append(dict(epsilon=epsilon, steps=steps, null_VFD=score, endpoint_KL=0.,
                 endpoint_paired_max_error=endpoint_error, FM_excess_loss=excess,
                 FM_excess_integration_error=integration_error, continuity_residual=max_continuity_residual,
                 analytic_flow_derivative_error=max_derivative_error,
                 genuine_mean_shift_VFD=shift_score, genuine_mean_shift_KL=SPEC["shift"]**2/2,
                 ranks_null_above_shift=score > shift_score))
    return dict(spec=SPEC, rows=rows,
        warning="Constructed approximate fields, not distinct exact FM population optima. Does not refute ideal-path identity or empirical VLA results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    root.mkdir(parents=True, exist_ok=False)
    result = audit()
    (root/"runner_source.py").write_bytes(Path(__file__).read_bytes())
    with (root/"RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (root/"MANIFEST.json").open("x", encoding="utf-8") as f:
        json.dump({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}, f, indent=2)
    print(json.dumps(result))
