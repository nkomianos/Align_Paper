#!/usr/bin/env python3
"""Build a sanitized, source-and-aggregate artifact for the memory paper."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "release" / "memory_boundary_iclr2027_artifact"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".tex", ".bib", ".sty", ".bst", ".txt"}
REDACTIONS = {
    "/home/ubuntu/align_research_20260910/": "${RESEARCH_ROOT}/",
    str(ROOT) + "\\": "${RESEARCH_ROOT}\\",
    str(ROOT).replace("\\", "\\\\") + "\\\\": "${RESEARCH_ROOT}\\\\",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def add(selected: set[Path], path: Path) -> None:
    if path.is_file():
        selected.add(path.resolve())


def sanitized_copy(source: Path, destination: Path) -> bool:
    """Copy one file, redacting local roots while preserving source hashes."""
    if source.suffix.lower() not in TEXT_SUFFIXES:
        shutil.copy2(source, destination)
        return False
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(source, destination)
        return False
    sanitized = text
    for private, portable in REDACTIONS.items():
        sanitized = sanitized.replace(private, portable)
    destination.write_text(sanitized, encoding="utf-8", newline="\n")
    return sanitized != text


def main() -> None:
    selected: set[Path] = set()
    for pattern in ("*.tex", "*.bib", "*.sty", "*.bst", "README.md"):
        for path in (ROOT / "paper_memory_boundary").glob(pattern):
            add(selected, path)
    for pattern in ("*.pdf", "*.png", "*.tex", "EVIDENCE.json"):
        for path in (ROOT / "paper_memory_boundary" / "generated").glob(pattern):
            add(selected, path)
    for path in (ROOT / "src" / "conditional_memory").rglob("*.py"):
        add(selected, path)
    for path in (ROOT / "scripts").glob("*memory_graft*.py"):
        add(selected, path)
    for name in ("build_memory_boundary_paper_evidence.py", "build_memory_boundary_release.py"):
        add(selected, ROOT / "scripts" / name)
    for path in (ROOT / "configs").glob("memory_graft*.json"):
        add(selected, path)
    for path in (ROOT / "docs").glob("MEMORY_GRAFT_SECURITY*.md"):
        add(selected, path)
    for name in ("MEMORY_GRAFT_SECURITY_PAPER_CLAIM_LEDGER_20260911.md",
                 "ICLR_2027_MEMORY_BOUNDARY_SUBMISSION_READINESS_20260911.md"):
        add(selected, ROOT / "docs" / name)
    for path in (ROOT / "tests").glob("*memory_graft*.py"):
        add(selected, path)

    evidence_names = {
        "DECISION.json", "DECISIVE.json", "DECISIVE_SUMMARY.json",
        "S2B_SUMMARY.json", "S2C_SUMMARY.json", "S2D_SUMMARY.json",
        "REPORT.json", "MANIFEST.json", "COMPLETE",
    }
    artifacts = ROOT / "artifacts"
    if artifacts.exists():
        for path in artifacts.rglob("*"):
            lower = path.name.lower()
            relevant = "memory_graft_security" in path.relative_to(artifacts).as_posix()
            if path.is_file() and relevant and (path.name in evidence_names or "verification" in lower):
                if path.stat().st_size <= 20 * (1 << 20):
                    add(selected, path)
    add(selected, artifacts / "memory_graft_security_s2e" /
        "memory_graft_security_s2e_quality_audit.json")
    add(selected, artifacts / "memory_graft_security_g6_pooling_audit.json")
    add(selected, artifacts / "memory_graft_clean_contribution_audit.json")
    add(selected, artifacts / "memory_graft_clean_contribution_audit_replay.json")
    add(selected, artifacts / "memory_graft_clean_contribution_verification.json")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = {}
    for source in sorted(selected):
        relative = source.relative_to(ROOT)
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        was_sanitized = sanitized_copy(source, destination)
        manifest[relative.as_posix()] = {
            "sha256": digest(destination),
            "source_sha256": digest(source),
            "bytes": destination.stat().st_size,
            "sanitized": was_sanitized,
        }
    readme = OUT / "ARTIFACT_README.md"
    readme.write_text(
        "# Addressing Is Not a Security Boundary — artifact\n\n"
        "This sanitized package contains paper source, frozen protocols and "
        "receipts, implementation and verifier code, tests, aggregate results, "
        "source inventories, figures, and replay reports. Model weights, raw "
        "corpora, SSH details, and per-prompt files are omitted from this compact "
        "bundle. The repository audit trail records their hashes and the paper "
        "states which claims rely on aggregate versus prediction-exact replay.\n",
        encoding="utf-8",
    )
    manifest["ARTIFACT_README.md"] = {
        "sha256": digest(readme), "bytes": readme.stat().st_size,
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
    print(json.dumps({"directory": str(OUT), "archive": archive,
                      "files": len(manifest), "archive_sha256": digest(Path(archive))},
                     indent=2))


if __name__ == "__main__":
    main()
