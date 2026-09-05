"""Validate and preserve the audit only; never invoke experiments or read TEST data."""
from collections import Counter
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/independent_audit_20260905"
START = "1cfe22124955dee56c8683cbf0ea0f51137e146a"
MODIFIED = {"README.md", "docs/RESEARCH_JOURNAL.md", "docs/ICLR_2027_SUBMISSION_READINESS.md", "docs/HINDSIGHT_GPU_QUEUE_RUNBOOK_20260904.md"}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    index_path = ROOT / "docs/RESEARCH_EVIDENCE_INDEX_20260905.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index["records"]
    assert len(records) == len({r["id"] for r in records}) == 144
    measured = Counter(r["classification"] for r in records if r["record_type"] == "measured_assay")
    assert measured == {"Valid negative": 11, "Invalid assay/capability": 21, "Developmental/apparatus only": 32}
    for source, digest in index["source_fingerprints"].items():
        assert sha(ROOT / source) == digest, source
    required = {"claim", "evidence_root", "protocol_commit", "unit_of_analysis", "verifier_result", "classification", "error_or_confound", "corrected_conclusion", "next_action"}
    assert all(required <= r.keys() for r in records)
    docs = sorted((ROOT / "docs").glob("*20260905.md"))
    docs += [ROOT / "docs/ICLR_2027_SUBMISSION_READINESS.md"]
    link_checks = []
    for doc in docs:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#")):
                continue
            path = target.strip("<>").split("#", 1)[0]
            if not path:
                continue
            assert (doc.parent / path).exists(), (doc.name, target)
            link_checks.append([doc.relative_to(ROOT).as_posix(), target])
    archive = ROOT / "docs/archive/ICLR_2027_SUBMISSION_READINESS_PRE_AUDIT_20260905.md"
    archive_hash = sha(archive)
    assert archive_hash == "e1458405c8ef4401c6fb8b5ade4633da71dd9279278f2d3a89c34db1d2d4e025"
    before = git("show", f"{START}:docs/ICLR_2027_SUBMISSION_READINESS.md")
    assert before.replace(b"\r\n", b"\n") == archive.read_bytes().replace(b"\r\n", b"\n")
    for filename in ("docs/RESEARCH_JOURNAL.md", "docs/HINDSIGHT_GPU_QUEUE_RUNBOOK_20260904.md"):
        historical_body = git("show", f"{START}:{filename}").decode("utf-8").replace("\r\n", "\n").split("\n", 2)[2]
        assert historical_body in (ROOT / filename).read_text(encoding="utf-8")
    changes = git("diff", "--name-status", START).decode("utf-8").splitlines()
    for change in changes:
        status, filename = change.split("\t", 1)
        assert status == "A" or filename in MODIFIED, change
    diff_check = subprocess.run(["git", "diff", "--check", START], cwd=ROOT, capture_output=True, text=True)
    assert diff_check.returncode == 0, diff_check.stdout + diff_check.stderr
    head = git("rev-parse", "HEAD").decode().strip()
    receipt = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "audited_starting_commit": START,
        "packaging_head": head,
        "scope": "Audit schema, source fingerprints, local document links, preserved journal/runbook body and readiness archive; no model rerun or confirmation parsing",
        "record_count": len(records), "measured_classifications": measured,
        "source_fingerprints_verified": len(index["source_fingerprints"]),
        "local_document_links_verified": len(link_checks),
        "archive_sha256": archive_hash,
        "historical_protocol_code_unchanged": True,
        "diff_check_exit_code": diff_check.returncode,
    }
    (AUDIT / "final_audit_qa.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if "--verify-only" in sys.argv:
        print(json.dumps(receipt, indent=2))
        return
    excluded = {"frozen_source_worktree", "__pycache__", ".pytest_cache"}
    files = [p for p in AUDIT.rglob("*") if p.is_file() and not (set(p.relative_to(AUDIT).parts) & excluded) and p.name != "AUDIT_PACKAGE_MANIFEST.json"]
    files += docs + [index_path, archive]
    files += [ROOT / filename for filename in MODIFIED]
    files += sorted((ROOT / "scripts").glob("*audit*20260905.py"))
    files = sorted(set(files))
    manifest = {
        "packaging_head": head, "excluded_directories": sorted(excluded),
        "scope": "Local audit evidence and reports only; historical raw archives remain at original paths. This is not a complete research-data export.",
        "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files},
    }
    manifest_path = AUDIT / "AUDIT_PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = ROOT / "artifacts/deployment" / f"independent_audit_evidence_{head[:7]}.zip"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files + [manifest_path]:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as bundle:
        assert bundle.testzip() is None
        for name, metadata in manifest["files"].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == metadata["sha256"], name
    digest = sha(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps({"qa": receipt, "archive": str(output), "archive_files": len(files) + 1, "archive_bytes": output.stat().st_size, "archive_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
