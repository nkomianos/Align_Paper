"""Verify the constructed Gaussian control using an analytic excess-loss formula."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_SHA = "6667401402df384a3368b9f18a353a206cbf71ff42872be9b913c4409644c6f0"


def verify(root):
    manifest = json.loads((root/"MANIFEST.json").read_text())
    if set(manifest) != {"runner_source.py", "RESULT.json"} or {p.name for p in root.iterdir()} != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("manifest coverage mismatch")
    for name, digest in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("checksum mismatch")
    if manifest["runner_source.py"] != SOURCE_SHA:
        raise ValueError("unexpected source")
    result = json.loads((root/"RESULT.json").read_text())
    rows = result["rows"]
    grid = {(e, n) for e in (0., .01, .05, .1) for n in (10, 50, 100, 500, 1000, 10000)}
    if len(rows) != 24 or {(r["epsilon"], r["steps"]) for r in rows} != grid:
        raise ValueError("incomplete grid")
    for r in rows:
        e, n = r["epsilon"], r["steps"]
        expected_excess = e**2*(1+2*np.pi**2/3)
        if not np.isclose(r["FM_excess_loss"], expected_excess, atol=1e-12):
            raise ValueError("excess-risk integral mismatch")
        t = np.arange(n)/n
        c = t*t+(1-t)**2
        expected_vfd = float(np.sum(t/(1-t)*2*c*(e*np.pi*np.cos(np.pi*t))**2)/n)
        positive = float(np.sum(t*(1-t)/c**2)/n*.25**2)
        if not np.isclose(r["null_VFD"], expected_vfd, atol=1e-12) or not np.isclose(r["genuine_mean_shift_VFD"], positive, atol=1e-12):
            raise ValueError("quadrature mismatch")
        if r["endpoint_KL"] != 0 or r["genuine_mean_shift_KL"] != .03125 or r["ranks_null_above_shift"] != (expected_vfd > positive):
            raise ValueError("control/rank mismatch")
        if r["endpoint_paired_max_error"] > 1e-12 or r["continuity_residual"] > 1e-12 or r["analytic_flow_derivative_error"] > 1e-8:
            raise ValueError("ODE control failed")
    return dict(scope="VERIFIED_EXACT_CONSTRUCTION_NOT_NATURAL_MODEL_EVIDENCE", paper_green_light=False,
        manifest_sha256=hashlib.sha256((root/"MANIFEST.json").read_bytes()).hexdigest(),
        grid_cells=24, closed_form_FM_excess="epsilon^2 * (1 + 2*pi^2/3)",
        example=[r for r in rows if r["epsilon"] == .05 and r["steps"] in (10, 100, 1000)],
        limitation="The analytic construction is not a failure rate on naturally trained flow models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    with args.report.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result))
