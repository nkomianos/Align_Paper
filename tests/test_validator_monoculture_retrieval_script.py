from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "retrieve_validator_monoculture_g0.ps1"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_retriever_has_frozen_inputs_and_fresh_destination_guards() -> None:
    source = _source()
    for parameter in (
        "$HostName",
        "$UserName",
        "$KeyPath",
        "$RemoteRoot",
        "$DestinationParent",
    ):
        assert parameter in source
    assert "RemoteRoot must be an absolute POSIX path" in source
    assert "RemoteRoot may not be the remote filesystem root" in source
    assert "Assert-NoReparseAncestor" in source
    assert "[System.IO.FileAttributes]::ReparsePoint" in source
    assert "[Guid]::NewGuid()" in source
    assert "Refusing existing exact retrieval target" in source
    assert "New-Item -ItemType Directory" in source


def test_retriever_uses_only_read_only_remote_operations() -> None:
    source = _source()
    assert 'Get-Command ssh.exe -CommandType Application' in source
    assert 'Get-Command scp.exe -CommandType Application' in source
    assert '"test", "-d", $RemoteRoot, "-a", "-f", $remoteManifest' in source
    assert '& $ssh.Source @sshArguments' in source
    assert '"sha256sum", "--", $remoteManifest' in source
    assert "Completion manifest changed during retrieval" in source
    assert '& $scp.Source @scpArguments' in source
    assert '"-r"' in source

    # Retrieval must never clean, move, chmod, or execute a remote shell payload.
    for forbidden in (
        "Remove-Item",
        "Move-Item",
        "Rename-Item",
        "rm -",
        "chmod ",
        "chown ",
        "Invoke-Command",
    ):
        assert forbidden not in source


def test_retriever_requires_completion_and_verifies_the_closed_file_set() -> None:
    source = _source()
    assert "COMPLETION_MANIFEST.json" in source
    assert "generation_complete__offline_analysis_pending" in source
    assert "$manifest.artifacts_sha256.PSObject.Properties" in source
    assert "Get-FileHash -LiteralPath" in source
    assert "SHA-256 mismatch" in source
    assert "Retrieved evidence contains a reparse point" in source
    assert "Retrieved evidence contains an unlisted file" in source
    assert "Write-Output $rootFull" in source
    assert "RETRIEVAL_RECEIPT.json" in source
    assert "evidence_root_sha256" in source
