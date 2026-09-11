import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("engram_g0_1", ROOT / "scripts" / "run_engram_addressability_g0_1.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CFG = json.loads((ROOT / "configs" / "engram_addressability_g0_preregistered.json").read_text())


def test_amendment_is_exact():
    amendment = json.loads((ROOT / "configs" / "engram_addressability_g0_1_amendment.json").read_text())
    MODULE.validate_amendment(amendment)
    amendment["single_authorized_change"]["calibration_epochs"] = 3
    try:
        MODULE.validate_amendment(amendment)
    except ValueError:
        pass
    else:
        raise AssertionError("modified amendment accepted")


def test_training_partition_is_disjoint():
    model = MODULE.base.build_model(CFG, "bigram")
    MODULE.set_trainable(model, "non_memory")
    assert all(not parameter.requires_grad for parameter in model.memory.parameters())
    MODULE.set_trainable(model, "memory")
    assert all(parameter.requires_grad for parameter in model.memory.parameters())
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("memory."))
