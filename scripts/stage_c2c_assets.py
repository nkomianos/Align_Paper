"""Stage pinned public C2C assets; download/checksum only, no inference."""
import argparse
import hashlib
import json
from pathlib import Path

FUSER = ("nics-efc/C2C_Fuser", "f01fc3258b305e280e04c7238f4f2cf31b7dc70d")
RECEIVER = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
PREFIX = "qwen3_0.6b+qwen3_4b_Fuser/final/"


def selected(filename, role):
    if role == "fuser":
        return filename.startswith(PREFIX) and filename.endswith((".json", ".pt"))
    return "/" not in filename and filename.endswith((".json", ".safetensors", ".txt", ".jinja"))


def digest_file(path, sibling):
    algorithm = "sha256" if sibling.lfs else "sha1"
    expected = sibling.lfs.sha256 if sibling.lfs else sibling.blob_id
    digest = hashlib.new(algorithm)
    if not sibling.lfs:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected or path.stat().st_size != sibling.size:
        raise ValueError(f"size/digest mismatch: {sibling.rfilename}")
    return {"name": sibling.rfilename, "bytes": sibling.size,
            "digest_kind": "sha256" if sibling.lfs else "git_blob_sha1", "digest": expected}


def stage(output):
    from huggingface_hub import HfApi, snapshot_download
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = {"scope": "DOWNLOAD_ONLY_NO_MODEL_EXECUTION", "gpu_calls": 0, "assets": {}}
        for role, (repo, revision) in (("receiver", RECEIVER), ("fuser", FUSER)):
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError("revision mismatch")
            files = [s for s in info.siblings if selected(s.rfilename, role)]
            if not files or (role == "fuser" and sum(s.rfilename.endswith(".pt") for s in files) != 28):
                raise ValueError("unexpected release inventory")
            root = Path(snapshot_download(repo, revision=revision, allow_patterns=[s.rfilename for s in files], max_workers=4))
            checks = [digest_file(root / s.rfilename, s) for s in files]
            result["assets"][role] = {"repo": repo, "revision": revision, "snapshot": str(root), "files": checks}
            (output / f"{role}_verified.json").write_text(json.dumps(result["assets"][role], indent=2), encoding="utf-8")
        (output / "COMPLETE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({role: {"snapshot": a["snapshot"], "files": len(a["files"])}
                          for role, a in result["assets"].items()}))
    except BaseException as exc:
        (output / "FAILED.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        raise
    finally:
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        (output / "MANIFEST.json").write_text(json.dumps({"files": hashes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    stage(parser.parse_args().output)
