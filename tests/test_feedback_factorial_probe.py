import numpy as np
import torch

from interaction_sprint.factorial_probe import SPEC, analyze, design, feedback_channel, teacher_prompt


def test_single_intent_factor_and_same_feedback_between_templates():
    data = design()
    assert len(data) == 16
    for source in {c["source_id"] for c in data}:
        a = next(c for c in data if c["source_id"] == source and c["intent"] == "stated")
        b = next(c for c in data if c["source_id"] == source and c["intent"] == "removed")
        assert a["prompt"].split("\n", 1)[1] == b["prompt"]
        assert a["options"] == b["options"]
        for o in (0, 1):
            suffix = f"Thanks, I prefer {a['options'][o]}."
            assert teacher_prompt(a, "released", o).endswith(suffix)
            assert teacher_prompt(a, "plain", o).endswith(suffix)


def test_factor_grid_null_and_real_opposite_parameter_directions():
    rows, jac = [], {}
    for c in design():
        rows.append({"id": c["id"], "intent": c["intent"], "task": c["task"], "answer": c["answer"],
                     "base": {"probabilities": [.9, .1]},
                     "teachers": {t: [{"probabilities": [.9878048780487805, .012195121951219513]},
                                      {"probabilities": [.5, .5]}] for t in SPEC["template"]}})
        jac[c["id"]] = torch.tensor([1., 2.])
    result = analyze(rows, jac)
    assert len(result["cells"]) == 24
    for cell in result["cells"]:
        if cell["rho"] == 0:
            assert cell["gradient_difference_norm"] < 1e-12
        if cell["reference"] == "fair" and cell["rho"] > 0:
            assert abs(cell["cosine"] + 1) < 1e-12
        assert len(cell["leave_one_task_out"]) == 4
    for ref in SPEC["feedback_reference"]:
        k = feedback_channel(1, .5, ref)
        assert (k > 0).all() and np.allclose(k.sum(1), 1)
