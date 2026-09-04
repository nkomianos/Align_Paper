from interaction_sprint.agentuq_inventory import auc


def test_auc_orientation_ties_missing():
    assert auc([{"failure": 0, "s": 1}, {"failure": 1, "s": 2}], "s")["auroc"] == 1
    assert auc([{"failure": 0, "s": 2}, {"failure": 1, "s": 1}], "s")["auroc"] == 0
    assert auc([{"failure": 0, "s": 2}, {"failure": 1, "s": 2}, {"failure": 1, "s": None}], "s") == {"n": 2, "failures": 1, "auroc": .5}
    assert auc([{"failure": 1, "s": 1}], "s")["auroc"] is None


def test_artifact_replay_and_tamper(tmp_path):
    import json
    import pytest
    from interaction_sprint.agentuq_inventory import analyze
    from interaction_sprint.cluster_certificate_audit import sha
    from scripts.verify_agent_monitor_audits import verify
    data, root = tmp_path/"data", tmp_path/"result"
    data.mkdir(); root.mkdir()
    simulation = {"task_id": "one", "trial": 0, "seed": 12,
                  "reward_info": {"reward": 1}, "messages": [], "uq_summary": {}}
    for i in range(6):
        (data/f"cell{i}.json").write_text(json.dumps({"simulations": [simulation]}), encoding="utf8")
    result = root/"RESULT.json"
    result.write_text(json.dumps(analyze(data)), encoding="utf8")
    (root/"MANIFEST.json").write_text(json.dumps({"RESULT.json": sha(result)}), encoding="utf8")
    assert verify(root, data, "agentuq")["verified"]
    result.write_text("{}", encoding="utf8")
    with pytest.raises(AssertionError):
        verify(root, data, "agentuq")
