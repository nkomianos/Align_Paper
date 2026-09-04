from pathlib import Path

import pytest

from scripts.sdpo_format_verifier_common_v3 import load_apparatus
from scripts.verify_sdpo_format_control_v3 import APPARATUS_SHA, CHECKER_SHA


def test_v3_replays_archived_checker_without_current_package_import(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src" / "interaction_sprint"
    (tmp_path / "apparatus_source.py").write_bytes((source / "sdpo_format_control_v3_data.py").read_bytes())
    (tmp_path / "checker_source.py").write_bytes((source / "sdpo_format_control_data.py").read_bytes())
    apparatus = load_apparatus(tmp_path, APPARATUS_SHA, CHECKER_SHA)
    rows = apparatus["build"]()["calibration"]
    assert len(rows) == 32
    for row in rows:
        assert apparatus["score"](apparatus["reference"](row), row)["joint"]
        assert "unit-example" not in str(row["prompt"])
    (tmp_path / "checker_source.py").write_text("# changed checker")
    with pytest.raises(ValueError, match="dependency differs"):
        load_apparatus(tmp_path, APPARATUS_SHA, CHECKER_SHA)
