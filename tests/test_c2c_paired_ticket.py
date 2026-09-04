import hashlib
import json

import pytest

from scripts.run_c2c_paired_update import SPEC, validate_ticket


def fixture(tmp_path):
    root = tmp_path / "update"
    merged = root / "merged_sender"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")
    (root / "MANIFEST.json").write_text("{}")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    entries = {str(seed): {"seed": seed, "manifest_sha256": digest(root / "MANIFEST.json"),
               "qualification": {"seed": seed, "decision": "QUALIFIED_FOR_PAIRED_INTERFACE_MEASUREMENT"},
               "merged_files": {"config.json": digest(merged / "config.json")}}
               for seed in (202609041, 202609042)}
    ticket = {"scope": "RELEASE_PAIRED_DEV_ONLY_NOT_PAPER_EXPANSION", "updates": entries,
              "final_cases_sha256": SPEC["cases_sha256"]}
    path = tmp_path / "ticket.json"
    def save():
        path.write_text(json.dumps(ticket))
        return digest(path)
    return root, merged, path, ticket, save


def test_ticket_locks_seed_and_merged_model_bytes(tmp_path):
    root, merged, path, ticket, save = fixture(tmp_path)
    sha = save()
    assert validate_ticket(path, sha, root, 202609041) == ticket
    (merged / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError):
        validate_ticket(path, sha, root, 202609041)


@pytest.mark.parametrize("kind", ["one_seed", "failed", "wrong_digest", "extra_file"])
def test_invalid_release_rejected(tmp_path, kind):
    root, merged, path, ticket, save = fixture(tmp_path)
    if kind == "one_seed":
        del ticket["updates"]["202609042"]
    elif kind == "failed":
        ticket["updates"]["202609042"]["qualification"]["decision"] = "UPDATE_NOT_QUALIFIED_NO_INTERFACE_CONCLUSION"
    elif kind == "extra_file":
        (merged / "extra").write_text("unexpected")
    sha = save()
    with pytest.raises(ValueError):
        validate_ticket(path, "0"*64 if kind == "wrong_digest" else sha, root, 202609041)
