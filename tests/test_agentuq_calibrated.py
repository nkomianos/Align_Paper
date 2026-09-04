import numpy as np
import pytest

from interaction_sprint.agentuq_calibrated import crossfit, task_folds


def test_fold_is_order_independent():
    ids = [str(i) for i in range(20)]
    assert dict(zip(ids, task_folds(ids))) == dict(zip(ids[::-1], task_folds(ids[::-1])))


def test_crossfit_task_is_not_used_in_training():
    ids = [str(i) for i in range(30)]
    scores = np.tile([[0, 1], [1, 0]], (15, 1))
    y = 1-scores
    result = crossfit(ids, scores, y)
    assert result["selector_success"] == 1
    for fold in result["fits"]:
        assert not set(fold["train_ids"]) & set(fold["test_ids"])
    changed = y.copy()
    changed[0] = 1-changed[0]
    again = crossfit(ids, scores, changed)
    assert result["per_task"][0]["predicted"] == again["per_task"][0]["predicted"]
    assert result["per_task"][0]["baseline_predicted"] == again["per_task"][0]["baseline_predicted"]


def test_invalid_labels():
    with pytest.raises(ValueError):
        crossfit(["a"]*10, [[0, 1]]*10, [[0, 1]]*10)
    with pytest.raises(ValueError):
        crossfit([str(i) for i in range(10)], [[0, 1]]*10, [[.5, 1]]*10)
