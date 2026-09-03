import json

import numpy as np
import pytest

from efference_pair.pilot import (CONDITIONS, FRAMES, MODEL_ID, MODEL_REVISION, SETTINGS, SIZE, decompose,
                                  dump, oracle_fields, render_conditions, scene,
                                  seal, time_labels, validate_seal)
from efference_pair.runner import planned_cases, token_contract
from efference_pair.verify import analyze, parse_answer, verify


@pytest.mark.parametrize("camera,obj", [(0, 0), (0, 1), (2, 3), (4, 1), (3, 4), (4, 4)])
def test_flow_and_coordinate_signs(camera, obj):
    frames, bg, rel, masks = scene(camera, obj)
    raw, g, r, transforms, diagnostics = decompose(frames)
    assert raw.shape == (15, SIZE, SIZE, 2)
    assert not any(d["failed"] for d in diagnostics)
    assert np.linalg.norm(np.median(g[5], axis=(0, 1)) - bg) < .35
    assert np.linalg.norm(np.median(r[5][masks[5]], axis=0) - rel) < .5
    assert np.allclose(raw, g + r, atol=1e-6)
    # Camera left + object left yields stationary object in image coordinates.
    if camera == obj == 0:
        assert np.linalg.norm(np.median(raw[5][masks[5]], axis=0)) < .2


def test_render_contract_and_control_anchors():
    frames, bg, rel, masks = scene(0, 2)
    raw, g, r, t, _ = decompose(frames)
    views = render_conditions(frames, raw, g, r, t, oracle_fields(bg, rel, masks), (-g, -r))
    assert set(views) == set(CONDITIONS)
    for condition, canvases in views.items():
        assert len(canvases) == FRAMES
        assert all(x.shape == (SIZE, SIZE, 3) and x.dtype == np.uint8 for x in canvases)
        if condition != "native_rgb":
            assert np.array_equal(canvases[:8], frames[::2])
    assert not np.array_equal(views["joint"][8:], views["sign_reverse"][8:])


def test_determinism():
    a, *_ = scene(2, 1)
    b, *_ = scene(2, 1)
    assert np.array_equal(a, b)
    assert np.array_equal(decompose(a)[1], decompose(b)[1])


def test_manifest_detects_changes_and_extras(tmp_path):
    dump(tmp_path / "x.json", {"a": 1})
    (tmp_path / "nested").mkdir()
    dump(tmp_path / "nested/MANIFEST.json", {"must_be_bound": True})
    seal(tmp_path)
    validate_seal(tmp_path)
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert "nested/MANIFEST.json" in manifest["files"]
    with pytest.raises(FileExistsError):
        dump(tmp_path / "x.json", {})
    (tmp_path / "extra").touch()
    with pytest.raises(ValueError, match="extra"):
        validate_seal(tmp_path)


def test_parser_is_not_posthoc_semantic_rescue():
    assert parse_answer(" B\n") == "B"
    assert parse_answer("```\nB\n```") is None
    assert parse_answer("B because...") is None


def test_budget_contract_and_mixed_resolution_rejection():
    inputs = {"image_grid_thw": np.array([[1, 14, 14]] * 16), "input_ids": np.array([[99] * 784 + [1, 2]])}
    assert token_contract(inputs, 99)["vision_tokens"] == 784
    inputs["image_grid_thw"][0, 1] = 28
    with pytest.raises(ValueError, match="resolution"):
        token_contract(inputs, 99)


def test_time_stamps_expose_nonchronological_layout():
    for c in CONDITIONS:
        assert len(time_labels(c)) == 16
    assert time_labels("rgb_layout")[7:9] == ["time 14", "time 01"]
    assert time_labels("joint")[8:10] == ["time 00 -> 01", "time 00 -> 01"]


def mock_records():
    cases, answers, records = [], {}, []
    for i in range(25):
        for stratum in ("camera", "object"):
            for condition in CONDITIONS:
                cid = f"scene-{i:03d}/{stratum}/{condition}"
                cases.append(dict(case_id=cid, scene=f"scene-{i:03d}", stratum=stratum, condition=condition))
                answers[cid] = dict(answer="A", reversed_answer="B", directional=i % 5 != 4,
                                    static=i % 5 == 4, object_group=i % 5)
                records.append(dict(case_id=cid, completion="A"))
    return cases, answers, records


def test_ceiling_is_not_a_paper_pass():
    c, a, r = mock_records()
    report = analyze(c, a, r)
    assert report["decision"] == "STOP_EP0_CEILING_NOT_INFORMATIVE"
    assert report["contrasts"]["joint_minus_native_rgb"]["independent_world_groups"] == 5


def test_missing_duplicate_outputs_fail_closed():
    c, a, r = mock_records()
    with pytest.raises(ValueError):
        analyze(c, a, r[:-1])
    with pytest.raises(ValueError):
        analyze(c, a, r[:-1] + [r[0]])


def test_smoke_not_full_result():
    c, a, r = mock_records()
    subset = planned_cases(c, "smoke")
    assert len(subset) == 24
    ids = {x["case_id"] for x in subset}
    report = analyze(subset, {k: v for k, v in a.items() if k in ids}, [x for x in r if x["case_id"] in ids])
    assert report["decision"] == "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"


def test_failure_is_not_concealed_by_perfect_oracle():
    c, a, r = mock_records()
    for row in r:
        if not row["case_id"].endswith("/oracle_joint"):
            row["completion"] = "B"
    assert analyze(c, a, r)["decision"] == "STOP_EP0_NO_FULL_BENCHMARK_SPEND"


def test_read_only_verifier_end_to_end_with_explicit_mock_evidence(tmp_path):
    cases, answers, records = mock_records()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
    dump(inputs / "answer_key.json", answers)
    dump(inputs / "settings.json", SETTINGS)
    plan = planned_cases(cases, "smoke")
    ids = {c["case_id"] for c in plan}
    records = [r for r in records if r["case_id"] in ids]
    contract = dict(vision_tokens=784, image_grid_thw=[[1, 14, 14]] * 16, total_input_tokens=900)
    for r in records:
        r.update(contract, seconds=1.0)
    dump(tmp_path / "budget_audit.json", {r["case_id"]: contract for r in records})
    dump(tmp_path / "plan.json", {"mode": "smoke", "case_ids": [c["case_id"] for c in plan]})
    dump(tmp_path / "runtime.json", {"model_id": MODEL_ID, "resolved_revision": MODEL_REVISION})
    dump(tmp_path / "COMPLETE.json", {"records": 24, "mode": "smoke"})
    (tmp_path / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    before = seal(tmp_path)
    report = verify(tmp_path)
    assert report["decision"] == "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
    assert validate_seal(tmp_path) == before
    with (tmp_path / "raw.jsonl").open("a") as f:
        f.write("tampered")
    with pytest.raises(ValueError, match="integrity"):
        verify(tmp_path)
