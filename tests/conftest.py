from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def smoke_config():
    from under_extinction.config import load_config

    return load_config(PROJECT_ROOT / "configs" / "smoke.yaml")


@pytest.fixture
def tiny_config(smoke_config):
    config = copy.deepcopy(smoke_config)
    config["data"].update({
        "train_examples": 48,
        "dev_examples": 24,
        "ordinary_worlds_per_renderer": 4,
        "audit_worlds_per_renderer": 3,
        "direct_conflict_worlds_per_renderer": 3,
        "comprehension_worlds_per_renderer": 2,
        "audit_comprehension_worlds_per_renderer": 1,
    })
    config["evaluation"]["bootstrap_replicates"] = 50
    return config
