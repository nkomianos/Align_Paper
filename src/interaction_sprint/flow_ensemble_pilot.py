"""Finite-data Gaussian flow ensemble pilot, without injected perturbations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

SPEC = dict(schema="natural-linear-flow-ensemble-v1", seeds=list(range(9046601, 9046609)),
            sizes=[32, 128, 512, 2048], degree=5, augmentation=4, ridge=1e-6,
            quadrature=[10, 100, 1000], endpoint_samples=64, query_labels=64,
            query_contexts=4, ode_rtol=1e-9, ode_atol=1e-11, paper_green_light=False,
            scope="CPU_GAUSSIAN_AFFINE_FLOW_TRAINING_NOT_VLA_REPLICATION")

TARGETS = [dict(name="isotropic", mean=[0., 0.], cov=[[1., 0.], [0., 1.]]),
           dict(name="shift", mean=[.5, -.5], cov=[[1., 0.], [0., 1.]]),
           dict(name="anisotropic", mean=[0., 0.], cov=[[.09, 0.], [0., 4.]]),
           dict(name="correlated", mean=[-.5, .5], cov=[[.25, .6], [.6, 2.25]])]


def basis(t):
    t = np.asarray(t)
    degree = SPEC["degree"]
    return np.stack([comb(degree, k)*t**k*(1-t)**(degree-k) for k in range(degree+1)], axis=-1)


def fit(targets, seed):
    rng = np.random.default_rng(seed)
    # Bootstrap the same finite target dataset for each ensemble member.
    labels = targets[rng.integers(len(targets), size=len(targets)*SPEC["augmentation"])]
    source = rng.normal(size=labels.shape)
    t = rng.uniform(size=len(labels))
    xt = (1-t[:, None])*source+t[:, None]*labels
    features = (basis(t)[:, :, None] * np.column_stack([xt, np.ones(len(xt))])[:, None, :]).reshape(len(xt), -1)
    y = labels-source
    gram = features.T@features/len(features)
    coefficients = np.linalg.solve(gram+SPEC["ridge"]*np.eye(gram.shape[0]), features.T@y/len(features))
    return coefficients.reshape(SPEC["degree"]+1, 3, 2)


def field_parts(coefficients, times):
    value = np.einsum("...k,kij->...ij", basis(times), coefficients)
    return np.swapaxes(value[..., :2, :], -1, -2), value[..., 2, :]


def solve(coefficients, rtol=None):
    def derivative(t, state):
        a, b = field_parts(coefficients, t)
        f, m = state[:4].reshape(2, 2), state[4:]
        return np.concatenate([(a@f).ravel(), a@m+b])
    sol = solve_ivp(derivative, (0., 1.), np.r_[np.eye(2).ravel(), np.zeros(2)], method="DOP853",
                    rtol=SPEC["ode_rtol"] if rtol is None else rtol, atol=SPEC["ode_atol"], dense_output=True)
    if not sol.success or not np.isfinite(sol.y).all():
        raise ValueError("ODE failure")
    return sol


def moments(solution, times):
    state = solution.sol(times).T
    f = state[..., :4].reshape(np.shape(times)+(2, 2))
    m = state[..., 4:]
    return m, f@np.swapaxes(f, -1, -2)


def gaussian_kl(m1, c1, m2, c2):
    sign1, log1 = np.linalg.slogdet(c1); sign2, log2 = np.linalg.slogdet(c2)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("nonpositive covariance")
    delta = m2-m1
    return float(.5*(np.trace(np.linalg.solve(c2, c1))+delta@np.linalg.solve(c2, delta)-2+log2-log1))


def vfd(coefficients, solutions, steps):
    t = np.arange(steps)/steps
    a1, b1 = field_parts(coefficients[0], t)
    a2, b2 = field_parts(coefficients[1], t)
    da, db = a1-a2, b1-b2
    total = np.zeros(steps)
    for sol in solutions:
        m, c = moments(sol, t)
        mean = np.einsum("nij,nj->ni", da, m)+db
        expected = np.einsum("nij,njk,nik->n", da, c, da)+(mean*mean).sum(1)
        total += expected/2
    return float(np.mean(t/(1-t)*total))


def energy(x, y):
    n, m = len(x), len(y)
    return float(2*cdist(x, y).mean()-cdist(x, x).sum()/(n*(n-1))-cdist(y, y).sum()/(m*(m-1)))


def evaluate(coefficients, target, seed):
    sols = [solve(c) for c in coefficients]
    final = [moments(s, np.asarray(1.)) for s in sols]
    m, c = np.asarray(target["mean"]), np.asarray(target["cov"])
    risk = float(np.mean([gaussian_kl(mm, cc, m, c) for mm, cc in final]))
    disagreement = .5*(gaussian_kl(*final[0], *final[1])+gaussian_kl(*final[1], *final[0]))
    rng = np.random.default_rng(seed)
    samples = [rng.multivariate_normal(mm, cc, SPEC["endpoint_samples"]) for mm, cc in final]
    # Common-source map difference: analytic expectation, not distribution-invariant.
    f1, f2 = [s.sol(1.)[:4].reshape(2, 2) for s in sols]
    common = float(((f1-f2)**2).sum()+((final[0][0]-final[1][0])**2).sum())
    return dict(risk_kl=risk, exact_ensemble_symmetric_kl=disagreement,
                endpoint_energy=energy(*samples), common_source_L2=common,
                **{f"vfd_{n}": vfd(coefficients, sols, n) for n in SPEC["quadrature"]}), sols


def summary(rows):
    pools = []
    scores = ["vfd_10", "vfd_100", "vfd_1000", "endpoint_energy", "common_source_L2", "exact_ensemble_symmetric_kl"]
    for seed in SPEC["seeds"]:
        group = [r for r in rows if r["seed"] == seed]
        if len(group) != 16:
            raise ValueError("incomplete seed pool")
        rng = np.random.default_rng(seed+777)
        selection = {}
        for score in scores+["random", "oracle_future_gain"]:
            if score == "random":
                chosen = rng.choice(len(group), SPEC["query_contexts"], replace=False).tolist()
            else:
                key = "risk_gain" if score == "oracle_future_gain" else score
                chosen = sorted(range(len(group)), key=lambda i: (-group[i][key], group[i]["id"]))[:SPEC["query_contexts"]]
            selection[score] = dict(ids=[group[i]["id"] for i in chosen],
                                  pool_risk_reduction=float(sum(group[i]["risk_gain"] for i in chosen)/len(group)))
        pools.append(dict(seed=seed, selections=selection,
              rank_vfd10_to_1000=float(spearmanr([r["vfd_10"] for r in group], [r["vfd_1000"] for r in group]).statistic),
              correlations_to_risk={s: float(spearmanr([r[s] for r in group], [r["risk_kl"] for r in group]).statistic) for s in scores},
              correlations_to_ensemble_kl={s: float(spearmanr([r[s] for r in group], [r["exact_ensemble_symmetric_kl"] for r in group]).statistic) for s in scores}))
    gains = {s: [p["selections"][s]["pool_risk_reduction"] for p in pools] for s in scores+["random", "oracle_future_gain"]}
    energy_beats = sum(a>b for a, b in zip(gains["endpoint_energy"], gains["vfd_10"]))
    means = {s: float(np.mean(g)) for s, g in gains.items()}
    ratio = means["endpoint_energy"]/means["vfd_10"] if means["vfd_10"] > 0 else None
    lead = energy_beats >= 6 and (ratio is not None and ratio >= 1.2)
    return dict(pools=pools, mean_pool_risk_reduction=means, energy_beats_vfd10_pools=energy_beats,
                energy_to_vfd10_gain_ratio=ratio, paper_green_light=False,
                decision="NATURAL_GAUSSIAN_ACQUISITION_LEAD_REQUIRES_REPLICATION" if lead else
                         "NO_PREFROZEN_PRACTICAL_LEAD_IN_GAUSSIAN_PILOT")


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (root/name).open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False)
    start = time.time()
    write("spec.json", SPEC); write("targets.json", TARGETS)
    (root/"runner_source.py").write_bytes(Path(__file__).read_bytes())
    rows, archive = [], {}
    try:
        for seed in SPEC["seeds"]:
            for ti, target in enumerate(TARGETS):
                for n in SPEC["sizes"]:
                    cell_seed = seed*10000+ti*3000+n
                    rng = np.random.default_rng(cell_seed)
                    all_data = rng.multivariate_normal(target["mean"], target["cov"], n+SPEC["query_labels"])
                    old = np.stack([fit(all_data[:n], cell_seed+k+10) for k in (0, 1)])
                    new = np.stack([fit(all_data, cell_seed+k+10) for k in (0, 1)])
                    before, sols = evaluate(old, target, cell_seed+20)
                    after, _ = evaluate(new, target, cell_seed+20)
                    cid = f"{seed}_{target['name']}_{n}"
                    # One stricter solve per cell is a numerical-error control, not a different predictor.
                    tight = solve(old[0], rtol=1e-11)
                    error = float(np.max(np.abs(tight.sol(1.)-sols[0].sol(1.))))
                    if error > 1e-6:
                        raise ValueError("solver accuracy check failed")
                    rows.append(dict(id=cid, seed=seed, target=target["name"], n=n, **before,
                             after_risk_kl=after["risk_kl"], risk_gain=before["risk_kl"]-after["risk_kl"], solver_endpoint_error=error))
                    archive[cid+"_data"] = all_data
                    archive[cid+"_before"] = old; archive[cid+"_after"] = new
        np.savez_compressed(root/"fitted_fields_and_data.npz", **archive)
        write("rows.json", rows)
        result = summary(rows)
        result["elapsed_seconds"] = time.time()-start
        result["fit_count"] = 512
        write("RESULT.json", result)
        write("COMPLETE.json", dict(contexts=128, fitted_fields=512, query_labels_per_pool=256))
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
        write("MANIFEST.json", manifest)
        print(json.dumps({k: v for k, v in result.items() if k != "pools"}), flush=True)
    except Exception as exc:
        write("FAILED.json", dict(exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(p.parse_args().root)
