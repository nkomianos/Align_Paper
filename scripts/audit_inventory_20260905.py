"""Read-only metadata inventory; never parses evidence or locked task contents."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/independent_audit_20260905"
SKIP = {".git", ".pytest_cache", "__pycache__", "independent_audit_20260905"}
SKIP_PREFIXES = (".venv",)


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return {"exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    directories = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(SKIP_PREFIXES)
                         and not (Path(base) == ROOT and d == "analysis"))
        relative = Path(base).relative_to(ROOT).as_posix()
        directories.append(relative)
        for name in sorted(names):
            p = Path(base) / name
            st = p.stat()
            rel = p.relative_to(ROOT).as_posix()
            low = name.lower()
            roles = []
            for role, hit in {
                "protocol_or_runbook": any(x in low for x in ("protocol", "runbook", "prereg")),
                "result_or_summary": any(x in low for x in ("result", "summary", "report", "decision")),
                "manifest_or_receipt": any(x in low for x in ("manifest", "sha256", "checksum", "receipt")),
                "checkpoint": any(x in low for x in ("checkpoint", "adapter", "optimizer")) or p.suffix.lower() in {".pt", ".safetensors"},
                "log": p.suffix.lower() in {".log", ".out", ".err"} or any(x in low for x in ("stderr", "stdout")),
                "bundle": p.suffix == ".bundle",
                "verifier": "verify" in low,
            }.items():
                if hit:
                    roles.append(role)
            files.append({"path": rel, "bytes": st.st_size, "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(), "roles": roles})
    roles = collections.Counter(r for x in files for r in x["roles"])
    roots = collections.defaultdict(lambda: {"files": 0, "bytes": 0, "roles": collections.Counter()})
    for row in files:
        parts = row["path"].split("/")
        key = "/".join(parts[:2]) if parts[0] in {"artifacts", "retrieved", "results", "deployment", "research_audits"} and len(parts) > 2 else parts[0]
        roots[key]["files"] += 1
        roots[key]["bytes"] += row["bytes"]
        roots[key]["roles"].update(row["roles"])
    bundles = []
    for row in files:
        if "bundle" not in row["roles"]:
            continue
        digest = hashlib.file_digest((ROOT / row["path"]).open("rb"), "sha256").hexdigest()
        bundles.append({"path": row["path"], "sha256": digest, "verification": git("bundle", "verify", row["path"]), "heads": git("bundle", "list-heads", row["path"])})
    receipt = {"audit_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exclusions": sorted(SKIP) + list(SKIP_PREFIXES) + ["analysis (repository root only)"], "content_policy": "Filesystem metadata only except reading deployment bundles for SHA256/git integrity. No confirmation/test JSON/keys parsed.", "git": {"head": git("rev-parse", "HEAD"), "status": git("status", "--short"), "branch": git("branch", "--show-current"), "divergence_local_tracking_ref_only": git("rev-list", "--left-right", "--count", "@{upstream}...HEAD"), "tags": git("tag", "--list")}, "file_count": len(files), "directory_count": len(directories), "roles": roles, "roots": roots, "bundles": bundles}
    for name, data in (("inventory.json", {"files": files, "directories": directories}), ("inventory_receipt.json", receipt)):
        (OUT / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({"file_count": len(files), "directory_count": len(directories), "roles": roles, "root_count": len(roots), "bundles": [{"path": b["path"], "sha256": b["sha256"], "exit_code": b["verification"]["exit_code"]} for b in bundles]}, indent=2))


if __name__ == "__main__":
    main()
