from pathlib import Path


def test_cpu_rehearsal_is_explicitly_non_substitutive():
    source = Path("scripts/run_hindsight_neural_gradient_cpu_dev.py").read_text(encoding="utf-8")
    protocol = Path("docs/HINDSIGHT_NEURAL_GRADIENT_CPU_REHEARSAL_PROTOCOL_20260904.md").read_text(
        encoding="utf-8"
    )
    assert "CPU_REHEARSAL_ONLY_NOT_NEURAL_G0" in source
    assert '"paper_green_light": False' in source
    assert "does not replace" in protocol
    assert "local_files_only=True" in source
