import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("engram_g0", ROOT / "scripts" / "run_engram_addressability_g0.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CFG = json.loads((ROOT / "configs" / "engram_addressability_g0_preregistered.json").read_text())


def test_design_is_deterministic_and_balanced():
    first = MODULE.validate_design(CFG)
    second = MODULE.validate_design(CFG)
    assert first == second
    assert len(MODULE.entity_map(CFG)) == 128
    assert sorted(MODULE.entity_map(CFG).values()).count(0) == 32


def test_address_control_is_parameter_matched_but_not_bigram_sensitive():
    MODULE.seed_everything(CFG["seed"])
    bigram = MODULE.build_model(CFG, "bigram")
    MODULE.seed_everything(CFG["seed"])
    control = MODULE.build_model(CFG, "current_token_control")
    assert MODULE.parameter_counts(bigram) == MODULE.parameter_counts(control)
    a = torch.tensor([[32, 48], [33, 48]])
    assert not torch.equal(bigram.memory.addresses(a)[0, -1], bigram.memory.addresses(a)[1, -1])
    assert torch.equal(control.memory.addresses(a)[0, -1], control.memory.addresses(a)[1, -1])


def test_trigger_random_and_benign_addresses_are_separate():
    MODULE.seed_everything(CFG["seed"])
    model = MODULE.build_model(CFG, "bigram")
    trigger = MODULE.pair_addresses(model, CFG["data"]["trigger_bigram"], "cpu")
    benign = MODULE.pair_addresses(model, CFG["data"]["benign_control_bigram"], "cpu")
    random_rows = MODULE.deterministic_random_addresses(CFG, [trigger, benign])
    for head in range(CFG["model"]["memory_heads"]):
        assert random_rows[head] not in {trigger[head], benign[head]}
