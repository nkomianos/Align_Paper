import pytest
import torch
from torch import nn

from latent_contract.cache_calibration import calibration_split, collect_bank, fit_layer, cache_pair


class TinyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, source, target):
        return tuple(t+self.projection(s) for s, t in zip(source, target))


def test_fit_check_selection_is_stratified_disjoint_and_order_invariant():
    cases = [{"case_id": f"{d}{i}", "dataset": d} for d in ("book", "arc") for i in range(64)]
    chosen = calibration_split(cases)
    assert len(chosen) == 96
    assert sum(cid.startswith("arc") for cid in chosen) == 48
    assert chosen == calibration_split(list(reversed(cases)))
    with pytest.raises(ValueError):
        calibration_split([{**c, "answer": "A"} for c in cases])


def test_layer_fit_preserves_all_methods_and_does_not_select_on_checks(tmp_path):
    torch.manual_seed(17)
    projector = TinyProjector().eval()
    bank = {}
    for part, n in (("fit", 100), ("check", 40)):
        old = {kind: torch.randn(1, 2, n, 4) for kind in ("key", "value")}
        bank[part] = {role+"_"+kind: (old[kind] if role == "old" else old[kind]*1.1 if role == "new" else torch.zeros_like(old[kind]))
                      for role in ("old", "new", "receiver") for kind in old}
    result = fit_layer(projector, bank, tmp_path / "fit", seed=10, steps=4)
    assert set(result["maps"]) == {"identity", "diagonal", "ridge", "orthogonal"}
    assert len(result["curve"]) == 4
    assert (tmp_path / "fit/retuned_bfloat16.pt").exists()
    assert result["maps"]["ridge"]["value"] < result["maps"]["identity"]["value"]
    assert result["selection"].startswith("none")


def test_bank_collects_exact_prefix_positions_and_separate_splits(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2, 3, 4, 5]
    config = Qwen3Config(vocab_size=20, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    models = {r: Qwen3ForCausalLM(config).eval() for r in ("old", "new", "receiver")}
    cases = [{"case_id": f"{d}{i}", "dataset": d, "question": "Q", "choices": list("ABCD")}
             for d in ("book", "arc") for i in range(3)]
    report = collect_bank(models, {r: Tokenizer() for r in models}, cases, lambda *a: "prompt",
                          [(0, 0), (1, 1)], tmp_path / "bank", fit_per_dataset=2, maximum_tokens=3)
    assert report["fit_cases"] == 4 and report["check_cases"] == 2
    assert report["backbone_forwards"] == 18
    assert report["cases"][0]["selected_prefix_positions"] == [0, 1, 3]
    bank = torch.load(tmp_path / "bank/layer_00.pt", weights_only=True)
    assert bank["fit"]["old_key"].shape == (1, 1, 12, 16)
    assert bank["check"]["old_key"].shape == (1, 1, 6, 16)
    with torch.inference_mode():
        reference = cache_pair(models["old"](torch.tensor([[1, 2, 3, 4]]), use_cache=True).past_key_values, 0)[0][:, :, [0, 1, 3], :]
    torch.testing.assert_close(bank["fit"]["old_key"][:, :, :3], reference)
