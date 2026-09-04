"""Read-only complete replay of the equal-count Gaussian acquisition control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from interaction_sprint import flow_fixed_count as pilot
from scripts.verify_flow_ensemble_pilot import close

SOURCE_SHA = "a5dafcdd8b0a9ab2a8f3200fb94f29e8422ee26ef4805a169f628216592e8b55"
SHARED_SHA = "975d8796e5b2a95a1606b00e6a98e862666072d318d451395ef8097654640377"


def verify(root):
    load = lambda name: json.loads((root/name).read_text())
    manifest = load("MANIFEST.json")
    if {p.name for p in root.iterdir()} != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("manifest coverage mismatch")
    for name, digest in manifest.items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("checksum mismatch")
    if manifest.get("runner_source.py") != SOURCE_SHA or manifest.get("shared_source.py") != SHARED_SHA:
        raise ValueError("source mismatch")
    if load("spec.json") != pilot.SPEC or load("targets.json") != pilot.targets():
        raise ValueError("frozen design mismatch")
    if load("COMPLETE.json") != dict(contexts=128, fitted_fields=512):
        raise ValueError("incomplete")
    rows = load("rows.json")
    expected = {f"{s}_{i}" for s in pilot.SPEC["seeds"] for i in range(16)}
    if len(rows) != 128 or {r["id"] for r in rows} != expected:
        raise ValueError("missing contexts")
    max_solver_error = 0.
    with np.load(root/"fields_and_data.npz", allow_pickle=False) as archive:
        if set(archive.files) != {cid+s for cid in expected for s in ("_before", "_after", "_data")}:
            raise ValueError("archive coverage mismatch")
        for r in rows:
            cid = r["id"]
            target = pilot.targets()[r["target_index"]]
            cell_seed = r["seed"]*100+r["target_index"]
            data = np.random.default_rng(cell_seed).multivariate_normal(target["mean"], target["cov"], 128)
            if not np.array_equal(data, archive[cid+"_data"]):
                raise ValueError("data replay mismatch")
            for phase, n in (("before", 64), ("after", 128)):
                coefficients = np.stack([pilot.shared.fit(data[:n], cell_seed+k+10) for k in (0, 1)])
                if not np.allclose(coefficients, archive[cid+"_"+phase], atol=1e-10, rtol=1e-8):
                    raise ValueError("fitting replay mismatch")
                values = pilot.evaluate(coefficients, target, cell_seed+20)
                if phase == "before" and not close(values, {k: r[k] for k in values}):
                    raise ValueError("score replay mismatch")
                if phase == "after" and not np.isclose(values["risk_kl"], r["after_risk_kl"], atol=1e-8):
                    raise ValueError("future risk mismatch")
                normal = pilot.shared.solve(coefficients[0]); strict = pilot.shared.solve(coefficients[0], rtol=1e-11)
                max_solver_error = max(max_solver_error, float(np.max(np.abs(normal.sol(1.)-strict.sol(1.)))))
            if not np.isclose(r["risk_gain"], r["risk_kl"]-r["after_risk_kl"], atol=1e-10):
                raise ValueError("gain mismatch")
    if max_solver_error > 1e-6:
        raise ValueError("solver sensitivity too large")
    result = load("RESULT.json"); replay = pilot.summarize(rows)
    if not close(replay, {k: result[k] for k in replay}):
        raise ValueError("decision/selection mismatch")
    return dict(scope="VERIFIED_FIXED_COUNT_GAUSSIAN_FIT_AND_SCORE_REPLAY", paper_green_light=False,
        **{k: v for k, v in replay.items() if k not in ("pools", "paper_green_light")},
        manifest_sha256=hashlib.sha256((root/"MANIFEST.json").read_bytes()).hexdigest(),
        maximum_solver_endpoint_error=max_solver_error,
        limitation="Same-code regeneration; Gaussian affine models, not neural or deployed VLA validation.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.root)
    with args.report.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result))
