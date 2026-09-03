import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("motion_sample", Path(__file__).parents[1] / "scripts/prepare_motionbench_dev_sample.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows, files = [], {}
    for category in module.CATEGORIES:
        for i in range(3):
            name = category.replace(" ", "_") + str(i) + ".mp4"
            rows.append({"question_type": category, "video_path": name,
                         "qa": [{"answer": "A" if i < 2 else None}],
                         "video_info": {"duration": 4}})
            files["MotionBench/self-collected/" + name] = {"size": 1234}
    return rows, files


def test_only_labelled_dev_and_deterministic():
    rows, files = fixture()
    chosen = module.choose(rows, files, 2)
    assert len(chosen) == 6
    assert chosen == module.choose(list(reversed(rows)), files, 2)
    assert all(item["annotation"]["qa"][0]["answer"] == "A" for item in chosen)


def test_no_hidden_test_or_external_path_fallback():
    rows, files = fixture()
    with pytest.raises(ValueError, match="insufficient"):
        module.choose(rows, files, 3)
    rows[0]["video_path"] = "../escape.mp4"
    with pytest.raises(ValueError, match="insufficient"):
        module.choose(rows, files, 2)


def test_no_silent_category_substitution():
    rows, files = fixture()
    rows[0]["video_type"] = "Gaming"
    with pytest.raises(ValueError, match="insufficient"):
        module.choose(rows, files, 2)
