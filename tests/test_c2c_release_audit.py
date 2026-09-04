import copy

import pytest

from scripts.audit_c2c_release import PREFIX, REVISION, inventory


def release():
    return {"sha": REVISION, "siblings": [
        {"rfilename": f"{PREFIX}/final/projector_{i}.pt", "size": 10,
         "lfs": {"sha256": "0" * 64}} for i in range(28)]}


def test_exact_release_inventory():
    result = inventory(release())
    assert result["bytes"] == 280
    assert result["projectors"] == 28
    assert result["weights_loaded"] is False


@pytest.mark.parametrize("mutation", ["revision", "missing", "size", "digest"])
def test_incomplete_or_unpinned_inventory_fails(mutation):
    data = copy.deepcopy(release())
    if mutation == "revision":
        data["sha"] = "main"
    elif mutation == "missing":
        data["siblings"].pop()
    elif mutation == "size":
        data["siblings"][0].pop("size")
    else:
        data["siblings"][0].pop("lfs")
    with pytest.raises(ValueError):
        inventory(data)
