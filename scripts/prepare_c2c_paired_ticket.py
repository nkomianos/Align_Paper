"""Local-only release of paired DEV after both sender updates qualify."""
import argparse
import json
from pathlib import Path

from scripts.run_c2c_sender_update import SPEC, sha256
from scripts.verify_c2c_sender_update import verify as verify_update
from scripts.verify_c2c_text_baseline import verify as verify_text


def prepare(roots, key, commit, text_root, baseline_root, prepared, upstream):
    comparator = verify_text(text_root, baseline_root, prepared, upstream)
    updates = {}
    for root in roots:
        report = verify_update(root, key, commit)
        if report["decision"] != "QUALIFIED_FOR_PAIRED_INTERFACE_MEASUREMENT":
            raise ValueError("both independent updates must qualify; do not select a favorable seed")
        seed = str(report["seed"])
        if seed in updates:
            raise ValueError("duplicate seed")
        manifest = json.loads((root / "MANIFEST.json").read_text())["files"]
        updates[seed] = {"seed": report["seed"], "manifest_sha256": sha256(root / "MANIFEST.json"),
                         "merged_files": {n[len("merged_sender/"):]: h for n, h in manifest.items() if n.startswith("merged_sender/")},
                         "qualification": report}
    if set(updates) != {str(s) for s in SPEC["seeds"]}:
        raise ValueError("missing fixed seed")
    return {"scope": "RELEASE_PAIRED_DEV_ONLY_NOT_PAPER_EXPANSION", "training_commit": commit,
            "updates": updates, "text_comparator": comparator, "text_manifest_sha256": sha256(text_root / "MANIFEST.json"),
            "final_cases_sha256": "66fca7bc7e5725f380a74db8454a1bc49f2304d7b5c267cd3ebadb965cc2e83f"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-root", type=Path, nargs=2, required=True)
    parser.add_argument("--training-commit", required=True)
    for name in ("key", "text-root", "baseline-root", "prepared", "upstream", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in [*args.update_root, args.text_root, args.baseline_root, args.prepared]):
        parser.error("ticket must be outside evidence")
    result = prepare(args.update_root, args.key, args.training_commit, args.text_root, args.baseline_root, args.prepared, args.upstream)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({"ticket_sha256": sha256(args.output), "seeds": sorted(result["updates"]), "scope": result["scope"]}))
