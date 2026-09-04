"""Read-only fitted-field, ODE and acquisition replay from saved evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from interaction_sprint.flow_ensemble_pilot import SPEC, TARGETS, fit, evaluate, summary

SOURCE_SHA = "975d8796e5b2a95a1606b00e6a98e862666072d318d451395ef8097654640377"


def close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return np.isclose(a, b, atol=1e-8, rtol=1e-7)
    return a == b


def verify(root):
    load = lambda name: json.loads((root/name).read_text())
    manifest = load("MANIFEST.json")
    if {p.name for p in root.iterdir()} != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("manifest coverage mismatch")
    for name, digest in manifest.items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("checksum mismatch")
    if manifest.get("runner_source.py") != SOURCE_SHA or load("spec.json") != SPEC or load("targets.json") != TARGETS:
        raise ValueError("frozen design mismatch")
    if load("COMPLETE.json") != dict(contexts=128, fitted_fields=512, query_labels_per_pool=256):
        raise ValueError("incomplete run")
    rows = load("rows.json"); original = load("RESULT.json")
    expected_ids = {f"{s}_{t['name']}_{n}" for s in SPEC["seeds"] for t in TARGETS for n in SPEC["sizes"]}
    if len(rows) != 128 or {r["id"] for r in rows} != expected_ids:
        raise ValueError("context coverage mismatch")
    with np.load(root/"fitted_fields_and_data.npz", allow_pickle=False) as archive:
        if set(archive.files) != {cid+suffix for cid in expected_ids for suffix in ("_data", "_before", "_after")}:
            raise ValueError("coefficient/data coverage mismatch")
        for r in rows:
            cid = r["id"]; n = r["n"]
            ti = next(i for i, t in enumerate(TARGETS) if t["name"] == r["target"])
            target = TARGETS[ti]
            cell_seed = r["seed"]*10000+ti*3000+n
            data = archive[cid+"_data"]
            expected_data = np.random.default_rng(cell_seed).multivariate_normal(target["mean"], target["cov"], n+64)
            if not np.array_equal(data, expected_data):
                raise ValueError("data regeneration mismatch")
            for phase, count in (("before", n), ("after", n+64)):
                fitted = np.stack([fit(data[:count], cell_seed+k+10) for k in (0, 1)])
                if not np.allclose(fitted, archive[cid+"_"+phase], atol=1e-10, rtol=1e-8):
                    raise ValueError("training replay mismatch")
                metrics, _ = evaluate(fitted, target, cell_seed+20)
                if phase == "before" and not close(metrics, {k: r[k] for k in metrics}):
                    raise ValueError("ODE/score replay mismatch")
                if phase == "after" and not np.isclose(metrics["risk_kl"], r["after_risk_kl"], atol=1e-8):
                    raise ValueError("post-query risk mismatch")
            if not np.isclose(r["risk_gain"], r["risk_kl"]-r["after_risk_kl"], atol=1e-10):
                raise ValueError("risk reduction mismatch")
    replay = summary(rows)
    if not close(replay, {k: original[k] for k in replay}):
        raise ValueError("selection/report mismatch")
    # Post-hoc diagnostic, explicitly not a replacement for the frozen criterion.
    count_baseline = []
    for seed in SPEC["seeds"]:
        group = [r for r in rows if r["seed"] == seed]
        chosen = sorted(group, key=lambda r: (r["n"], r["id"]))[:4]
        count_baseline.append(sum(r["risk_gain"] for r in chosen)/16)
    return dict(scope="VERIFIED_GAUSSIAN_FITTING_AND_ODE_REPLAY", paper_green_light=False,
        manifest_sha256=hashlib.sha256((root/"MANIFEST.json").read_bytes()).hexdigest(),
        decision=replay["decision"], mean_pool_risk_reduction=replay["mean_pool_risk_reduction"],
        energy_beats_vfd10_pools=replay["energy_beats_vfd10_pools"],
        energy_to_vfd10_gain_ratio=replay["energy_to_vfd10_gain_ratio"],
        median_vfd_resolution_rank_correlation=float(np.median([p["rank_vfd10_to_1000"] for p in replay["pools"]])),
        posthoc_smallest_dataset_baseline=dict(mean_risk_reduction=float(np.mean(count_baseline)), per_pool=count_baseline),
        limitations=["Replay uses committed fitting and scoring implementations, not an independently implemented learner.",
                     "Affine two-dimensional fields and Gaussian targets; not neural VLA evidence.",
                     "Different initial label counts make acquisition easier; smallest-dataset baseline is post-hoc.",
                     "No separate calibration or failure-detection task; no guarantee for other flow model classes."])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.root)
    with args.report.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result))
