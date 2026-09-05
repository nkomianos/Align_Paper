"""Extract safe runtime metadata from existing local receipts; no host access."""
from pathlib import Path
import collections
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/independent_audit_20260905"
ALLOW = {"gpu", "model", "model_id", "revision", "git_commit", "torch", "transformers", "cuda", "weight_updates", "wall_seconds", "elapsed_seconds", "device", "python"}


def main():
    inventory = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
    logs, runtimes, exits, bundles = [], [], [], []
    for row in inventory["files"]:
        relative = row["path"]
        parts = relative.split("/")
        if parts[0] not in {"retrieved", "artifacts"} or any("_source_" in p for p in parts):
            continue
        path = ROOT / relative
        if path.suffix == ".log" and row["bytes"] < 10_000_000:
            value = path.read_text(encoding="utf-8", errors="replace")
            logs.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tracebacks": value.count("Traceback (most recent call last)"), "exception_classes": sorted(set(re.findall(r"^([A-Za-z]+(?:Error|Exception)):", value, re.MULTILINE))), "has_cuda_reference": "CUDA" in value or "cuda" in value, "has_gh200_reference": "GH200" in value})
        elif path.suffix == ".exit":
            value = path.read_text(encoding="utf-8", errors="replace").strip()
            exits.append({"path": relative, "code": int(value) if re.fullmatch(r"-?\d+", value) else "not_an_integer_receipt"})
        elif path.name == "runtime.json":
            value = json.loads(path.read_text(encoding="utf-8"))
            runtimes.append({"path": relative, "receipt": {k:v for k,v in value.items() if k in ALLOW}})
        elif path.suffix == ".bundle" and path.parent.name == "deployment":
            sidecar = path.with_suffix(".bundle.sha256")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = sidecar.read_text(encoding="utf-8-sig").split()[0].lower() if sidecar.exists() else None
            bundles.append({"path": relative, "sha256": digest, "sidecar_present": sidecar.exists(), "sidecar_matches": digest == expected if expected else None})
    report = {"kind": "local receipt audit; no monitoring, process replay, billing or host contact", "logs": logs, "runtimes": runtimes, "exit_receipts": exits, "deployment_bundle_sidecars": bundles, "counts": {"log_paths": len(logs), "unique_log_contents": len({r["sha256"] for r in logs}), "log_paths_with_tracebacks": sum(r["tracebacks"]>0 for r in logs), "runtime_paths": len(runtimes), "exit_codes": dict(collections.Counter(r["code"] for r in exits))}, "limits": ["Copied logs, snapshots and archives are not independent experiments.", "Some failed launches lack exit/PID receipts; no comprehensive billing or utilization trace exists here.", "SSH timeout does not demonstrate process termination or an inactive paid host.", "No current GPU status is asserted."]}
    (OUT / "runtime_receipts.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "bundle_sidecars": bundles},indent=2))


if __name__ == "__main__":
    main()
