from pathlib import Path
import pytest
import torch

from interaction_sprint.sdpo_format_positive_control import (
    released_loss_class, released_reference_check, hindsight_messages,
    completion_logps, NativeBridge, calibration_decision, positive_decision)


def test_released_loss_actual_pinned_formula_and_gradient():
    path = Path(__file__).parents[1] / "artifacts/user_interactions_objective_audit_v1/online_sdpo_updater.py"
    if not path.exists():
        pytest.skip("pinned upstream archive not available locally")
    result = released_reference_check(released_loss_class(path.read_bytes()))
    assert result["gradient_max_error"] == 0
    assert result["teacher_gradient_is_absent"]
    with pytest.raises(ValueError):
        released_loss_class(b"not the pinned source")


def test_hindsight_is_exact_block_last_user_no_mutation():
    messages = [dict(role="user", content="first"), dict(role="assistant", content="reply"),
                dict(role="user", content="last")]
    output = hindsight_messages(messages, "  correction ")
    assert messages[-1]["content"] == "last"
    assert output[0] == messages[0]
    assert output[-1]["content"] == "last\n\n=== HINDSIGHT CONTEXT ===\n[The following is a future user message. Use this to guide your answer to the user prompt.]\ncorrection"


def test_fresh_generation_has_no_probability_distorting_processors():
    from types import SimpleNamespace
    from interaction_sprint.sdpo_format_positive_control import generation_config
    cfg = generation_config(SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1), True, 64)
    assert cfg.temperature == cfg.top_p == cfg.typical_p == cfg.repetition_penalty == 1
    assert cfg.top_k == 0 and cfg.pad_token_id == 0 and cfg.min_length == 0
    assert cfg.num_beams == 1
    for field in ("bad_words_ids", "forced_bos_token_id", "forced_eos_token_id", "suppress_tokens", "begin_suppress_tokens",
                  "sequence_bias", "watermarking_config", "guidance_scale", "min_p"):
        assert getattr(cfg, field, None) is None


def test_native_completion_bridge_matches_full_forward_all_actual_tokens():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(2)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=24, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()
    context, y = [1, 7, 8], [9, 6, 2]
    actual = completion_logps(model, context, y, "cpu", True)
    full = model(torch.tensor([context + y]), use_cache=False).logits[:, len(context)-1:-1].float()
    expected = full.log_softmax(-1).gather(-1, torch.tensor([y]).unsqueeze(-1)).squeeze(-1)
    assert actual.shape == (1, 3)
    assert torch.allclose(actual, expected, atol=1e-6)
    bridge = NativeBridge(model, "cpu", context, [1, 5, 4, 7])
    lp, mask, _ = bridge._compute_token_logprobs("teacher", torch.tensor([y]), need_grad=False)
    assert not lp.requires_grad
    assert mask.tolist() == [[1, 1, 1]]  # including actual terminal token; no padding mask
    assert len(bridge.saved["teacher"]) == 3


def scored(joint=True, content=True):
    return dict(score=dict(joint=joint, format_valid=joint, content_valid=content))


def test_qualification_requires_teacher_recovery_and_explicit_competence():
    originals = [scored(False)] * 16 + [scored()] * 16
    good = [scored()] * 32
    assert calibration_decision(good, good, originals)["qualified"]
    assert not calibration_decision(good, originals, originals)["qualified"]
    assert not calibration_decision(originals, good, originals)["qualified"]
    assert not positive_decision(good, good)["positive_control"]
    assert positive_decision(originals, good)["positive_control"]
