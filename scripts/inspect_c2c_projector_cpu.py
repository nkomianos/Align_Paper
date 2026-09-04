"""CPU-only load/shape check of one released C2C projector, not an LM test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote
from urllib.request import urlopen

UPSTREAM_COMMIT = "113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a"


def inspect(upstream, metadata, output):
    upstream = upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != UPSTREAM_COMMIT or dirty:
        raise ValueError("upstream checkout not the audited clean revision")
    manifest = json.loads((metadata / "MANIFEST.json").read_text())["files"]
    for name, expected in manifest.items():
        if Path(name).name != name or hashlib.sha256((metadata / name).read_bytes()).hexdigest() != expected:
            raise ValueError("metadata checksum mismatch")
    info = json.loads((metadata / "fuser_api.json").read_text())
    summary = json.loads((metadata / "summary.json").read_text())
    row = next(r for r in summary["selected_files"] if r["rfilename"].endswith("/final/projector_0.pt"))
    output.mkdir(parents=True, exist_ok=False)
    try:
        url = f"https://huggingface.co/nics-efc/C2C_Fuser/resolve/{info['sha']}/{quote(row['rfilename'])}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        (output / "projector_0.pt").write_bytes(data)
        if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["lfs"]["sha256"]:
            raise ValueError("released weight size/digest mismatch")
        import torch
        import transformers
        torch.set_num_threads(2)
        torch.manual_seed(20260904)
        sys.path.insert(0, str(upstream))
        from rosetta.model import projector as projector_module
        if not Path(projector_module.__file__).resolve().is_relative_to(upstream):
            raise ValueError("wrong upstream module imported")
        config = json.loads((metadata / "projector_0.json").read_text())
        if config["class"] != "C2CProjector" or config["init_args"]["dtype"] != {"__type__": "torch.dtype", "value": "bfloat16"}:
            raise ValueError("unexpected released projector config")
        proj = projector_module.load_projector(str(metadata / "projector_0.json")).eval()
        state = torch.load(output / "projector_0.pt", map_location="cpu", weights_only=True)
        proj.load_state_dict(state, strict=True)
        inputs = [torch.randn(1, 8, 4, 128, dtype=torch.bfloat16) for _ in range(4)]
        with torch.inference_mode():
            result = proj(tuple(inputs[:2]), tuple(inputs[2:]))
            repeated = proj(tuple(inputs[:2]), tuple(inputs[2:]))
        if any(r.shape != (1, 8, 4, 128) or not torch.isfinite(r).all() for r in result):
            raise ValueError("invalid projector forward shape/value")
        if not all(torch.equal(a, b) for a, b in zip(result, repeated)):
            raise ValueError("nondeterministic eval projector")
        report = {"scope": "ONE_PROJECTOR_CPU_ABI_CHECK_NOT_MODEL_REPRODUCTION", "gpu_calls": 0,
                  "upstream": revision, "source_url": url, "weight_sha256": row["lfs"]["sha256"],
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "published_dependency_match": False, "strict_state_dict_pass": True,
                  "finite_forward_pass": True, "deterministic_eval": True,
                  "key_gate_logit": float(proj.key_gate_logit.detach()),
                  "value_gate_logit": float(proj.value_gate_logit.detach())}
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    except BaseException as exc:
        (output / "FAILED.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        raise
    finally:
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        (output / "MANIFEST.json").write_text(json.dumps({"files": hashes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("upstream", "metadata", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    inspect(args.upstream, args.metadata, args.output)
