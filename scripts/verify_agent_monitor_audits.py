"""Read-only hash and exact analysis replay for two exploratory CPU audits."""
import argparse
import json
from pathlib import Path

from interaction_sprint.cluster_certificate_audit import sha, audit_data, exact_cluster_case
from interaction_sprint.agentuq_inventory import analyze


def verify(root, data, kind):
    manifest = json.loads((root/"MANIFEST.json").read_text(encoding="utf8"))
    assert manifest == {"RESULT.json": sha(root/"RESULT.json")}
    result = json.loads((root/"RESULT.json").read_text(encoding="utf8"))
    if kind == "agentuq":
        # JSON serializes integer histogram keys as strings.
        assert result == json.loads(json.dumps(analyze(data)))
    else:
        import interaction_sprint.cluster_certificate_audit as module
        assert result["source_sha256"] == sha(Path(module.__file__))
        assert result["source_data_sha256"] == {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()}
        assert result["data_audit"] == json.loads(json.dumps(audit_data(data)))
        for row in result["counterexamples"]:
            assert row == exact_cluster_case(row["tasks"], row["repeats"], row["true_recall"], row["target"], row["alpha"])
    return {"verified": True, "kind": kind, "manifest_sha256": sha(root/"MANIFEST.json"),
            "scope": "Source/data hashes and deterministic replay; not independent implementation or agent replication."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--kind", choices=["cluster", "agentuq"], required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    receipt = verify(args.root, args.data, args.kind)
    with args.receipt.open("x", encoding="utf8") as out:
        json.dump(receipt, out, indent=2)
    print(json.dumps(receipt))
