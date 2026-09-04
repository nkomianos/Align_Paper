"""Tiny native-model wrapper smoke, only under the pinned C2C Transformers API."""
import copy
from pathlib import Path
import sys

import pytest


def test_sender_switch_and_disabled_fuser_leave_no_cache_residue():
    import torch
    from torch import nn
    import transformers
    if transformers.__version__ != "4.52.4":
        pytest.skip("run in artifacts/c2c_cpu_compat_v1 for published C2C API")
    upstream = Path(__file__).resolve().parents[1] / "artifacts/c2c_upstream_audit_20260904"
    if not upstream.exists():
        pytest.skip("pinned upstream checkout unavailable")
    sys.path.insert(0, str(upstream))
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from rosetta.model.wrapper import RosettaModel
    from scripts.run_c2c_paired_update import generate_record

    class TinyTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1, 2, 3, 4]])

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids.tolist())

    class TinyFuser(nn.Module):
        def forward(self, source, target):
            return tuple(t + .05*s for s, t in zip(source, target))

    config = Qwen3Config(vocab_size=20, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=16, attention_dropout=0, eos_token_id=None, pad_token_id=0)
    torch.manual_seed(30)
    receiver = Qwen3ForCausalLM(config).eval()
    old = Qwen3ForCausalLM(copy.deepcopy(config)).eval()
    new = Qwen3ForCausalLM(copy.deepcopy(config)).eval()
    fused = RosettaModel([receiver, old], projector_list=[TinyFuser(), TinyFuser()]).eval()
    for layer in range(2):
        fused.set_projector_config(1, layer, 0, layer, layer)
    mapping = copy.deepcopy(fused.projector_dict)
    calls = {"old": 0, "new": 0}
    def hook(name):
        def count(module, inputs, output):
            calls[name] += 1
        return count
    handles = [old.register_forward_hook(hook("old")), new.register_forward_hook(hook("new"))]
    case = {"case_id": "tiny", "dataset": "synthetic"}
    turns = [{"role": "user", "content": "synthetic"}]
    tok = TinyTokenizer()
    with torch.inference_mode():
        original = generate_record(receiver, tok, turns, case, "receiver", "cpu", max_new_tokens=3)
        first = generate_record(fused, tok, turns, case, "old_c2c", "cpu", fused=True, max_new_tokens=3)
        fused.model_list[1] = new
        generate_record(fused, tok, turns, case, "new_c2c", "cpu", fused=True, max_new_tokens=3)
        fused.projector_dict = {}
        disabled = generate_record(fused, tok, turns, case, "new_disabled", "cpu", fused=True, max_new_tokens=3)
        fused.model_list[1] = old
        fused.projector_dict = mapping
        repeat = generate_record(fused, tok, turns, case, "old_c2c", "cpu", fused=True, max_new_tokens=3)
        after = generate_record(receiver, tok, turns, case, "receiver", "cpu", max_new_tokens=3)
    for handle in handles:
        handle.remove()
    assert calls["old"] > 0 and calls["new"] > 0
    assert first["generated_ids"] == repeat["generated_ids"]
    assert original["generated_ids"] == disabled["generated_ids"] == after["generated_ids"]
