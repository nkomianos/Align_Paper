"""Fresh-seed, equal-label-count acquisition control; no GPU or injected rotations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import flow_ensemble_pilot as shared

SPEC = dict(schema="flow-fixed-count-v1", seeds=list(range(9046801, 9046809)), target_seed=9046799,
            contexts=16, initial_labels=64, added_labels=64, query_contexts=4,
            shared_model_spec=shared.SPEC, paper_green_light=False)
SCORES = ["vfd_10", "vfd_100", "vfd_1000", "endpoint_energy", "whitened_energy",
          "sample_gaussian_kl", "common_source_L2", "exact_ensemble_symmetric_kl"]


def targets():
    rng = np.random.default_rng(SPEC["target_seed"])
    result = []
    for i in range(SPEC["contexts"]):
        angle = rng.uniform(0, np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        sigma = np.exp(rng.uniform(np.log(.5), np.log(2.), 2))
        result.append(dict(name=f"target{i:02d}", mean=rng.uniform(-.8, .8, 2).tolist(),
                           cov=(rotation@np.diag(sigma**2)@rotation.T).tolist()))
    return result


def sample_scores(x, y):
    pooled = np.cov(np.concatenate([x, y]), rowvar=False)
    eig, vec = np.linalg.eigh(pooled)
    if np.min(eig) <= 0:
        raise ValueError("degenerate sample covariance")
    whitening = vec @ np.diag(1/np.sqrt(eig)) @ vec.T
    m1, m2 = x.mean(0), y.mean(0)
    c1, c2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    kl = .5*(shared.gaussian_kl(m1, c1, m2, c2)+shared.gaussian_kl(m2, c2, m1, c1))
    return dict(whitened_energy=shared.energy(x@whitening, y@whitening), sample_gaussian_kl=kl)


def evaluate(coefficients, target, seed):
    values, sols = shared.evaluate(coefficients, target, seed)
    rng = np.random.default_rng(seed)
    samples = [rng.multivariate_normal(*shared.moments(s, np.asarray(1.)), shared.SPEC["endpoint_samples"]) for s in sols]
    return {**values, **sample_scores(*samples)}


def summarize(rows):
    pools = []
    for seed in SPEC["seeds"]:
        group = [r for r in rows if r["seed"] == seed]
        if len(group) != 16:
            raise ValueError("incomplete pool")
        results = {}
        for score in SCORES+["random", "oracle_future_gain"]:
            if score == "random":
                ids = np.random.default_rng(seed+777).choice(16, 4, replace=False)
            else:
                field = "risk_gain" if score == "oracle_future_gain" else score
                ids = sorted(range(16), key=lambda j: (-group[j][field], group[j]["id"]))[:4]
            results[score] = dict(ids=[group[j]["id"] for j in ids], gain=float(sum(group[j]["risk_gain"] for j in ids)/16))
        pools.append(dict(seed=seed, results=results))
    means = {s: float(np.mean([p["results"][s]["gain"] for p in pools])) for s in pools[0]["results"]}
    wins = sum(p["results"]["whitened_energy"]["gain"] > p["results"]["vfd_10"]["gain"] for p in pools)
    ratio = means["whitened_energy"]/means["vfd_10"] if means["vfd_10"] > 0 else None
    lead = wins >= 6 and ratio is not None and ratio >= 1.2
    return dict(pools=pools, mean_pool_risk_reduction=means, whitened_energy_wins=wins,
        whitened_to_vfd_ratio=ratio, paper_green_light=False,
        decision="FIXED_COUNT_ACQUISITION_LEAD_REQUIRES_REPLICATION" if lead else "NO_FIXED_COUNT_ACQUISITION_LEAD")


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (root/name).open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False)
    started = time.time()
    write("spec.json", SPEC); write("targets.json", targets())
    (root/"runner_source.py").write_bytes(Path(__file__).read_bytes())
    (root/"shared_source.py").write_bytes(Path(shared.__file__).read_bytes())
    rows, saved = [], {}
    try:
        for seed in SPEC["seeds"]:
            for i, target in enumerate(targets()):
                cell_seed = seed*100+i
                rng = np.random.default_rng(cell_seed)
                data = rng.multivariate_normal(target["mean"], target["cov"], 128)
                old = np.stack([shared.fit(data[:64], cell_seed+k+10) for k in (0, 1)])
                new = np.stack([shared.fit(data, cell_seed+k+10) for k in (0, 1)])
                before = evaluate(old, target, cell_seed+20)
                after = evaluate(new, target, cell_seed+20)
                cid = f"{seed}_{i}"
                rows.append(dict(id=cid, seed=seed, target_index=i, **before,
                           after_risk_kl=after["risk_kl"], risk_gain=before["risk_kl"]-after["risk_kl"]))
                saved[cid+"_data"] = data; saved[cid+"_before"] = old; saved[cid+"_after"] = new
        np.savez_compressed(root/"fields_and_data.npz", **saved)
        write("rows.json", rows)
        result = summarize(rows); result["elapsed_seconds"] = time.time()-started
        write("RESULT.json", result); write("COMPLETE.json", dict(contexts=128, fitted_fields=512))
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
        write("MANIFEST.json", manifest)
        print(json.dumps({k: v for k, v in result.items() if k != "pools"}), flush=True)
    except Exception as exc:
        write("FAILED.json", dict(exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(p.parse_args().root)
