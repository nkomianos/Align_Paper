from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_validator_monoculture_g0_remote.sh"


def test_remote_orchestrator_binds_a_unique_clean_frozen_runtime() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "secrets.token_hex(32)" in source
    assert '"remote_run_root": str(root)' in source
    assert 'export PYTHONPATH="$REPO_ROOT/src"' in source
    assert "export PYTHONNOUSERSITE=1" in source
    assert "export PYTHONSAFEPATH=1" in source
    assert 'sys.version_info[:2] != (3, 12)' in source
    assert 'torch.__version__.startswith("2.7.1")' in source
    assert 'str(torch.version.cuda) != "12.8"' in source
    assert "resolved_snapshot" not in source


def test_terminal_resume_preserves_link_window_temporaries() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'if [[ "$RESUME_MODE" == "1" && -f "$RUN_ROOT/COMPLETION_MANIFEST.json" ]]' in source
    assert ".terminal-recovery" in source
    assert "completed root has unlisted files beyond recoverable terminal temporaries" in source
