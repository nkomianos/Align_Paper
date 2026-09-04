import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.verify_sdpo_format_control import (qualification, paired_users, released_loss,
                                              verify_loss, validate_native_generation, hindsight, checkpoints)


def test_qualification_is_apparatus_not_paper_decision():
    def row(joint, content=True):
        return {"score": {"joint": joint, "content_valid": content, "format_valid": joint}}
    original = [row(False)] * 4
    explicit = [row(True)] * 4
    assert qualification(original, explicit, [row(True)] * 3 + [row(False)])["qualified"]
    assert not qualification(original, explicit, [row(True)] * 2 + [row(False)] * 2)["qualified"]


def test_hindsight_keeps_core_input_and_adds_only_future_feedback():
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    output = hindsight(messages, " feedback ")
    assert messages[-1]["content"] == "task"
    assert output[-1]["content"].endswith("feedback")
    assert output[0] == messages[0]


def test_pairing_uses_users_not_independent_tasks():
    before = [{"id": str(i), "user_id": "a" if i < 2 else "b", "score": {"joint": False}} for i in range(4)]
    after = [dict(r, score={"joint": r["user_id"] == "a"}) for r in before]
    result = paired_users(before, after)
    assert result["users"] == 2 and result["joint_gain_pp"] == 50
    with pytest.raises(ValueError):
        paired_users(before, after[:-1])


def test_native_emitted_eos_and_text_rejected_if_tampered():
    tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda ids, skip_special_tokens: "answer" if skip_special_tokens else "answer<EOS>")
    row = {"prompt_ids": [1], "completion_ids": [2, 9], "token_logprobs": [-1., -.1],
           "text": "answer", "raw_decoded": "answer<EOS>", "terminated": True,
           "sampling": False, "seed": 3, "elapsed_seconds": 1.}
    validate_native_generation(row, tokenizer, [1], 10, False, 3)
    broken = dict(row, completion_ids=[9, 2])
    with pytest.raises(ValueError, match="EOS"):
        validate_native_generation(broken, tokenizer, [1], 10, False, 3)
    with pytest.raises(ValueError, match="decoded"):
        validate_native_generation(dict(row, text="fabrication"), tokenizer, [1], 10, False, 3)


def test_upstream_hash_guard():
    with pytest.raises(ValueError, match="pinned"):
        released_loss(b"not the upstream code")


def test_actual_released_loss_replay_and_tamper():
    source = Path("artifacts/user_interactions_objective_audit_v1/online_sdpo_updater.py")
    if not source.exists():
        pytest.skip("pinned external source artifact not installed")
    cls = released_loss(source.read_bytes().replace(b"\r\n", b"\n"))
    base = torch.tensor([[-1., -2., -.5]], requires_grad=True)
    teacher = torch.tensor([[-.5, -3., -.1]])
    stub = SimpleNamespace(config=SimpleNamespace(signal_clip=0), _log_token_table=lambda *a: None)
    stub._compute_token_logprobs = lambda name, *a, **kw: (base if name == "base" else teacher, torch.ones_like(base, dtype=torch.long), None)
    loss, metrics = cls._simple_signal_loss(stub, "base", "teacher", torch.tensor([[1, 2, 3]]))
    row = {"response": {"completion_ids": [1, 2, 3]}, "training_logprobs": {"base": base.detach()[0].tolist(), "teacher": teacher[0].tolist()},
           "metrics": metrics, "loss": float(loss.detach()), "grad_norm": .2, "update_seconds": 1.}
    verify_loss(row, cls)
    broken = copy.deepcopy(row); broken["metrics"]["completion_tokens"] = 2
    with pytest.raises(ValueError):
        verify_loss(broken, cls)
    broken = copy.deepcopy(row); broken["loss"] += 1
    with pytest.raises(ValueError):
        verify_loss(broken, cls)


def test_optimizer_steps_and_adapter_keys(tmp_path):
    torch.save({"x.a": torch.zeros(2)}, tmp_path / "initial_adapter.pt")
    torch.save({"x.a": torch.ones(2)}, tmp_path / "final_adapter.pt")
    state = {"state": {0: {"step": torch.tensor(64.), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)}},
             "param_groups": [{"params": [0], "lr": .001, "eps": 1e-6, "weight_decay": 0}]}
    torch.save(state, tmp_path / "final_optimizer.pt")
    config = {"lr": .001, "eps": 1e-6, "weight_decay": 0}
    assert checkpoints(tmp_path, config)["squared_parameter_change"] == 2
    state["state"][0]["step"] = torch.tensor(63.)
    torch.save(state, tmp_path / "final_optimizer.pt")
    with pytest.raises(ValueError, match="update count"):
        checkpoints(tmp_path, config)
