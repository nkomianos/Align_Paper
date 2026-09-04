"""Pin public C2C release metadata without importing upstream code or weights.

Writes only to a new evidence directory. This is provenance preflight, not a
model reproduction, dependency compatibility test, or scientific gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REPO = "nics-efc/C2C_Fuser"
REVISION = "f01fc3258b305e280e04c7238f4f2cf31b7dc70d"
PREFIX = "qwen3_0.6b+qwen3_4b_Fuser"
MODELS = {
    "receiver": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "sender": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}


def inventory(metadata):
    if metadata["sha"] != REVISION:
        raise ValueError("fuser revision mismatch")
    files = [f for f in metadata["siblings"] if f["rfilename"].startswith(PREFIX + "/")]
    final = [f for f in files if f["rfilename"].startswith(PREFIX + "/final/")]
    weights = [f for f in final if f["rfilename"].endswith(".pt")]
    expected = {f"{PREFIX}/final/projector_{i}.pt" for i in range(28)}
    if {f["rfilename"] for f in weights} != expected:
        raise ValueError("unexpected projector inventory")
    for row in files:
        if not isinstance(row.get("size"), int) or row["size"] < 0:
            raise ValueError("missing file size")
    for row in weights:
        if len(row.get("lfs", {}).get("sha256", "")) != 64:
            raise ValueError("missing weight digest")
    return {"selected_files": files, "bytes": sum(f["size"] for f in files),
            "projectors": len(weights), "weights_loaded": False}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    urls = {}

    def fetch(name, url):
        with urlopen(url, timeout=60) as response:
            data = response.read()
        (output / name).write_bytes(data)
        urls[name] = url
        return json.loads(data)

    try:
        metadata = fetch("fuser_api.json", f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true")
        summary = inventory(metadata)
        base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(PREFIX)}"
        for name, path in (("training_config.json", "config.json"),
                           ("layer_mapping.json", "final/projector_config.json"),
                           ("projector_0.json", "final/projector_0.json")):
            fetch(name, f"{base}/{path}")
        models = {}
        for role, (repo, revision) in MODELS.items():
            info = fetch(f"{role}_api.json", f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
            if info["sha"] != revision:
                raise ValueError(f"{role} revision mismatch")
            config = fetch(f"{role}_config.json", f"https://huggingface.co/{repo}/resolve/{revision}/config.json")
            models[role] = {"repo": repo, "revision": revision, "layers": config["num_hidden_layers"],
                            "kv_heads": config["num_key_value_heads"], "hidden_size": config["hidden_size"]}
        summary.update({"scope": "METADATA_ONLY_NOT_A_REPRODUCTION", "models": models,
                        "fuser_revision": REVISION, "upstream_model_revisions_specified": False})
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "selected_files"}))
    finally:
        (output / "sources.json").write_text(json.dumps(urls, indent=2), encoding="utf-8")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        (output / "MANIFEST.json").write_text(json.dumps({"files": hashes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
