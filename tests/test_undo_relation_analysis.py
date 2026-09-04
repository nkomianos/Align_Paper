import pytest
import hashlib
import json
import random
from scripts.analyze_undo_relation_training import paired_difference, qualify, verify_budget, analyze


def test_pairing_preserves_disagreements():
    result = paired_difference({"a": True, "b": False}, {"a": False, "b": True})
    assert result["gain_pp"] == 0
    assert result["wins"] == result["losses"] == 1


def test_bad_pairing_rejected():
    with pytest.raises(ValueError):
        paired_difference({"a": True}, {"b": True})


def test_qualification_does_not_require_history_or_intervention_success():
    rows = [{"condition": condition, "prediction": "A", "target": "A" if condition in ("canonical", "padded", "counterfactual") else "B", "choice_mass": .8}
            for condition in ("canonical", "padded", "counterfactual", "history", "explicit_update")]
    assert qualify(rows)["qualified"]
    rows[0]["prediction"] = "B"
    assert not qualify(rows)["qualified"]


def test_matched_budget_and_tampered_indices(tmp_path):
    config = {"seed": 1, "epochs": 2, "batch_size": 1}
    rng = random.Random(1)
    plan = []
    for _ in range(2):
        order = [0, 1]; rng.shuffle(order)
        plan.extend([[i] for i in order])
    tokenized = {"student": {"train": [[1], [2, 3]]}, "schedule": plan,
                 "teacher": {"canonical_distillation": [[1], [2]], "local_relation": [[1], [2]]}}
    manifest = {}
    for name in ("initial_adapter.pt", "canonical_distillation_teacher.pt", "local_relation_teacher.pt"):
        (tmp_path / name).write_bytes(b"fixture-not-real-weights"); manifest[name] = "unused"
    for arm in ("terminal_sft", "canonical_distillation", "local_relation"):
        (tmp_path / arm).mkdir()
        for name in ("final_adapter.pt", "final_optimizer.pt"):
            (tmp_path / arm / name).write_bytes(b"fixture-not-real-weights"); manifest[f"{arm}/{name}"] = "unused"
        logs = [{"step": step, "indices": ids, "student_tokens": 1 + ids[0], "loss": 1., "grad_norm": 1., "elapsed_seconds": 1.} for step, ids in enumerate(plan)]
        (tmp_path / arm / "training.jsonl").write_text("\n".join(map(json.dumps, logs)))
        manifest[f"{arm}/training.jsonl"] = "unused"
    summary = {"training_steps_per_arm": 4, "student_tokens_per_arm": 6,
               "teacher_tokens": {"canonical_distillation": 2, "local_relation": 2}}
    assert verify_budget(tmp_path, manifest, {"arguments": config}, tokenized, summary)["schedule_reproduced"]
    tokenized["schedule"] = [[1], [1], [1], [1]]
    with pytest.raises(ValueError, match="schedule"):
        verify_budget(tmp_path, manifest, {"arguments": config}, tokenized, summary)


def test_unqualified_evidence_is_not_paper_kill(tmp_path):
    def write(name, obj):
        (tmp_path / name).write_text(json.dumps(obj))
    data = [{"id": c, "condition": c, "target": "B", "pair_id": "p", "depth": 20, "prompt": "x"}
            for c in ("canonical", "padded", "counterfactual")]
    for split in ("train", "dev", "eval"):
        write(f"{split}.json", data)
    (tmp_path / "runner_source.py").write_text("# fixture")
    (tmp_path / "lora_source.py").write_text("# fixture")
    h = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    write("freeze.json", {"input_sha256": {s: h(f"{s}.json") for s in ("train", "dev", "eval")},
                          "source_sha256": h("runner_source.py"), "lora_source_sha256": h("lora_source.py")})
    write("tokenized.json", {"student": {"dev": [[1], [1], [1]]}})
    rows = [dict(r, prediction="A", choice_probabilities=[.7, .1, .1, 0.], choice_mass=.9, prompt_tokens=1) for r in data]
    (tmp_path / "baseline_dev.jsonl").write_text("\n".join(map(json.dumps, rows)))
    write("qualification.json", qualify(rows))
    write("status.json", {"status": "UNQUALIFIED_ASSAY", "scientific_decision": None})
    write("MANIFEST.json", {p.name: h(p.name) for p in tmp_path.iterdir()})
    result = analyze(tmp_path)
    assert result["status"] == "UNQUALIFIED_ASSAY"
    assert result["scientific_decision"] is None
