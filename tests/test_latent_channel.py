import copy
import json

import pytest

from latent_contract.channel import (ARMS, analyze, plan, prepare, receiver_text,
                                    sender_text, validate)
from latent_contract.fixture import build


def perfect(cases, key):
    rows = []
    for case in cases:
        cid = case["case_id"]
        other = cid.rsplit("/", 1)[0] + ("/counterfactual" if cid.endswith("/original") else "/original")
        for arm in ARMS:
            target = other if arm.startswith("counterfactual_") else cid
            completion = chr(65 + case["choices"].index(key[target]))
            if arm == "no_message":
                completion = "A"
            rows.append({"case_id": cid, "arm": arm, "completion": completion})
    return rows


def test_receiver_blind_sender_changes():
    cases, _ = build()
    for a, b in zip(cases[::2], cases[1::2]):
        assert receiver_text(a) == receiver_text(b)
        assert sender_text(a) != sender_text(b)
        assert "Query item:" not in sender_text(a)
        assert all(code not in sender_text(a) for code in a["choices"])


def test_smoke_cannot_pass_thesis():
    cases, key = build()
    cases = plan(cases, "smoke")
    assert len(cases) * len(ARMS) == 48
    report = analyze(cases, key, perfect(cases, key), "smoke")
    assert report["decision"] == "SMOKE_ONLY_NO_THESIS_DECISION"
    assert report["updates_trained"] == 0


def test_full_formation_does_not_claim_updates():
    cases, key = build()
    report = analyze(cases, key, perfect(cases, key), "full")
    assert report["decision"] == "READY_TO_DESIGN_UPDATE_STUDY_NOT_PAPER_PASS"


def test_incompetence_is_invalid_not_hypothesis_kill():
    cases, key = build()
    rows = perfect(cases, key)
    for row in rows:
        if row["arm"] == "text":
            row["completion"] = "I cannot solve this."
    assert analyze(cases, key, rows, "full")["decision"] == "INVALID_CHANNEL_ASSAY"


def test_no_message_leakage_stops():
    cases, key = build()
    rows = perfect(cases, key)
    answers = {row["case_id"]: row["completion"] for row in rows if row["arm"] == "text"}
    for row in rows:
        if row["arm"] == "no_message":
            row["completion"] = answers[row["case_id"]]
    assert analyze(cases, key, rows, "full")["decision"] == "INVALID_CHANNEL_ASSAY"


def test_no_payload_dependence_parks_implementation():
    cases, key = build()
    rows = perfect(cases, key)
    for row in rows:
        if row["arm"] == "counterfactual_latent":
            case = next(c for c in cases if c["case_id"] == row["case_id"])
            row["completion"] = chr(65 + case["choices"].index(key[row["case_id"]]))
    assert analyze(cases, key, rows, "full")["decision"] == "PARK_THIS_CHANNEL_IMPLEMENTATION_NOT_UPDATE_HYPOTHESIS"


def test_duplicates_and_missing_stop():
    cases, key = build()
    rows = perfect(cases, key)
    for broken in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
        with pytest.raises(ValueError, match="grid"):
            analyze(cases, key, broken, "full")


def test_prepare_seal_and_no_overwrite(tmp_path):
    root = tmp_path / "inputs"
    prepare(root)
    validate(root)
    with pytest.raises(FileExistsError):
        prepare(root)
    (root / "cases.json").write_text(json.dumps([]))
    with pytest.raises(ValueError, match="checksum"):
        validate(root)


def test_alignment_finite_shapes_and_no_inplace_mutation():
    import torch
    from latent_contract.channel import alignment
    generator = torch.Generator().manual_seed(82)
    states = torch.randn(12, 8, generator=generator)
    reference = torch.randn(12, 8, generator=generator)
    vocabulary = torch.randn(40, 8, generator=generator)
    original = states.clone()
    result = alignment(states, reference, vocabulary)
    assert result.shape == states.shape
    assert torch.isfinite(result).all()
    assert torch.equal(states, original)
    assert torch.allclose(result, alignment(states, reference, vocabulary))
    with pytest.raises(ValueError, match="identical"):
        alignment(states[:-1], reference, vocabulary)


def test_tiny_random_qwen_inputs_embeds_contract():
    """Random tiny CPU model: integration shape test, no scientific accuracy."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                        eos_token_id=31, pad_token_id=0)
    model = Qwen3ForCausalLM(config).eval()
    tokens = torch.tensor([[1, 4, 8, 3, 6]])
    with torch.inference_mode():
        hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
        assert hidden.shape == (1, 5, 16)
        output = model.generate(inputs_embeds=model.get_input_embeddings()(tokens),
                                attention_mask=torch.ones_like(tokens), max_new_tokens=2,
                                do_sample=False)
    assert 1 <= output.shape[1] <= 2


def test_resealed_wrong_inputs_fail(tmp_path):
    from latent_contract.channel import seal, validate_inputs
    root = tmp_path / "inputs"
    prepare(root)
    (root / "cases.json").write_text("[]")
    (root / "MANIFEST.json").unlink()
    seal(root)
    with pytest.raises(ValueError, match="deterministic"):
        validate_inputs(root)


def test_full_offline_verifier_and_immutable_root(tmp_path):
    from latent_contract.channel import SETTINGS, seal, write
    from latent_contract.verify import verify
    root = tmp_path / "evidence"
    root.mkdir()
    prepare(root / "inputs")
    all_cases, key = build()
    cases = plan(all_cases, "smoke")
    rows = perfect(cases, key)
    for row in rows:
        row["input_tokens"] = 16 if row["arm"] == "no_message" else 24
    write(root / "runtime.json", {"model": SETTINGS["model"], "revision": SETTINGS["revision"]})
    write(root / "plan.json", {"settings": SETTINGS, "mode": "smoke", "arms": list(ARMS),
                              "case_ids": [c["case_id"] for c in cases],
                              "receiver_forwards": 48, "sender_prefills": 8})
    write(root / "BUDGET_AUDIT.json", [{"case_id": c["case_id"], "input_tokens_with_message": 24,
                                      "input_tokens_without_message": 16} for c in cases])
    write(root / "COMPLETE.json", {"records": 48, "updates_trained": 0})
    (root / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    seal(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert verify(root)["decision"] == "SMOKE_ONLY_NO_THESIS_DECISION"
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (root / "raw.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify(root)
